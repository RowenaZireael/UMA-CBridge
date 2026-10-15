#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import errno
import re
import sys
import argparse
import json
import logging
import multiprocessing as mp
import os
import queue as pyqueue
import signal
import socket
import struct
from dataclasses import dataclass
from contextlib import suppress
import time

import numpy as np
from ase import Atoms, units
from ase.build import molecule

# User settings
MODEL_PATH = ""  # Required: absolute path to your UMA checkpoint (.pt).
PORT = 50423  # Must match the wrappers' FASTUMA_PORT; unused by --ipi.
DEVICE = "cuda"  # "cuda" or "cpu"

# Default inference settings
INFERENCE_TF32 = False  # Faster GPU matrix multiplication with reduced precision.
INFERENCE_MERGE_MOLE = False  # Speed up inference by merging MoLE weights; requires fixed composition, charge and spin.
INFERENCE_COMPILE = False  # Speed up repeated inference with PyTorch compilation; adds first-run compilation time.

# --md and --ipi settings
MD_INFERENCE_TF32 = False
MD_INFERENCE_MERGE_MOLE = True
MD_INFERENCE_COMPILE = True

# Queue batching: enable with --batch; collects ready energy/gradient requests.
# i-PI uses XML batch_size instead; the supplied example enables batches of 32 without --batch.
# CUDA OOM: split the batch in half and retry; report an error if one job still fails.
RAW_BATCH_MAX_SIZE = 32  # Maximum requests per batch; no waiting to fill a batch.
RAW_BATCH_MAX_ATOMS = 2048  # Total atom budget per batch; larger individual structures run alone.

# Hessian finite differences and displacement batching
HESSIAN_STEP = 1e-3  # Central-difference displacement in Angstrom.
HESSIAN_BATCH_SIZE = 8  # Maximum displaced structures evaluated together.
HESSIAN_BATCH_MAX_ATOMS = 2048  # Total atom budget per displacement batch; at least one structure runs.
HESSIAN_BATCH_REDUCE_STEP = 2  # Reduce batch size by this amount on GPU OOM: 8 -> 6 -> 4 -> 2 -> 1.


# Wire protocol helpers
async def _read_exactly(reader: asyncio.StreamReader, n: int) -> bytes:
    return await reader.readexactly(n)

async def _recv_job(reader: asyncio.StreamReader) -> tuple[dict, bytes]:
    max_header_bytes = 65536
    max_payload_bytes = 64 * 1024 * 1024
    hb = await _read_exactly(reader, 8)
    hlen = struct.unpack("<Q", hb)[0]
    if hlen == 0 or hlen > max_header_bytes:
        raise ValueError(f"header length {hlen} outside allowed range")
    h = await _read_exactly(reader, hlen)
    meta = json.loads(h.decode("utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("header must be a JSON object")

    pb = await _read_exactly(reader, 8)
    plen = struct.unpack("<Q", pb)[0]
    if plen > max_payload_bytes:
        raise ValueError(f"payload length {plen} exceeds limit {max_payload_bytes}")
    p = await _read_exactly(reader, plen)
    return meta, p

def _raw_with_len(payload: dict | None, error: str | None = None) -> bytes:
    if error is not None:
        msg = error.encode("utf-8", "replace")
        data = struct.pack("<II", 1, len(msg)) + msg
        return struct.pack("<Q", len(data)) + data

    energy = float(payload["energy"])
    grad = payload.get("gradient")
    if grad is None:
        grad_arr = np.empty(0, dtype="<f8")
    else:
        grad_arr = np.asarray(grad, dtype="<f8").reshape(-1)

    hessian_lower = payload.get("hessian_lower")
    if hessian_lower is None:
        hess_arr = np.empty(0, dtype="<f8")
    else:
        hess_arr = np.asarray(hessian_lower, dtype="<f8").reshape(-1)

    data = (
        struct.pack("<IdI", 0, energy, grad_arr.size)
        + grad_arr.tobytes()
        + struct.pack("<I", hess_arr.size)
        + hess_arr.tobytes()
    )
    return struct.pack("<Q", len(data)) + data

def decode_wire_payload(payload: bytes, meta: dict) -> dict:
    """Decode a C-wrapper request for the model worker."""
    payload_format = str(meta["payload_format"]).lower()
    if payload_format != "fastuma_raw_v1":
        raise ValueError(f"unsupported payload_format: {payload_format}")
    if len(payload) < 4:
        raise ValueError("raw payload too short")
    natoms = struct.unpack_from("<I", payload, 0)[0]
    expected = 4 + 2 * natoms + 12 * natoms
    if len(payload) != expected:
        raise ValueError(f"raw payload length {len(payload)} != expected {expected}")
    z = np.frombuffer(payload, dtype="<u2", count=natoms, offset=4)
    bad_mask = (z < 1) | (z > 118)
    if np.any(bad_mask):
        bad = int(z[bad_mask][0])
        raise ValueError(f"unsupported atomic number: {bad}")
    positions = np.frombuffer(payload, dtype="<f4", count=3 * natoms, offset=4 + 2 * natoms).reshape((-1, 3))
    return {
        "atomic_numbers": z.astype(np.int16, copy=True),
        "positions": positions.copy(),
        "charge": int(meta["charge"]),
        "multiplicity": int(meta["multiplicity"]),
        "calc_type": str(meta["calc_type"]).lower(),
    }

# GPU worker process

# Unit conversions
ANG2BOHR = 1.0 / 0.529177249
EV2HARTREE = 0.036749322

def _build_calc(*, periodic_stress: bool = False):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    from fairchem.core.units.mlip_unit import load_predict_unit

    settings = InferenceSettings(
        tf32=INFERENCE_TF32,
        activation_checkpointing=False,
        merge_mole=INFERENCE_MERGE_MOLE,
        compile=INFERENCE_COMPILE,
        external_graph_gen=False,
        internal_graph_gen_version=2,
        **({"predict_untrained_stress": {"omol"}} if periodic_stress else {}),
    )
    predictor = load_predict_unit(path=MODEL_PATH, device=DEVICE, inference_settings=settings)
    return FAIRChemCalculator(predictor, task_name="omol")


def _energy(atoms) -> float:
    return float(atoms.get_potential_energy() * EV2HARTREE)


def _gradient(atoms) -> np.ndarray:
    forces_eVa = atoms.get_forces()
    grad = -forces_eVa * EV2HARTREE / ANG2BOHR
    return grad


def _displaced_forces(atoms, start: int, count: int, h: float, batch_size: int) -> np.ndarray:
    """Predict one block of +/- displacements without changing the ASE state."""
    from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

    calc = atoms.calc
    graphs = []
    for offset in range(batch_size):
        # Repeat the last real displacement to keep the tail's compiled shape.
        sample = start + min(offset, count - 1)
        displaced = atoms.copy()
        coordinate = sample // 2
        displaced.positions[coordinate // 3, coordinate % 3] += h if sample % 2 == 0 else -h
        graphs.append(calc.a2g(displaced))
    data = graphs[0] if batch_size == 1 else atomicdata_list_to_batch(graphs)
    prediction = calc.predictor.predict(data)
    forces = prediction['forces'].detach().cpu().numpy()
    if forces.shape != (batch_size * len(atoms), 3) or not np.isfinite(forces).all():
        raise ValueError('UMA returned invalid Hessian displacement forces')
    return forces.reshape(batch_size, 3 * len(atoms))[:count].astype(np.float64)


def _numeric_hessian(atoms, h: float = 1e-3) -> np.ndarray:
    """Central-difference Hessian in fixed-size blocks, reduced on CUDA OOM."""
    import torch

    if len(atoms) == 0 or not np.isfinite(h) or h <= 0:
        raise ValueError('Hessian requires nonempty atoms and a positive finite step')
    if min(HESSIAN_BATCH_SIZE, HESSIAN_BATCH_MAX_ATOMS, HESSIAN_BATCH_REDUCE_STEP) < 1:
        raise ValueError('Hessian batch size, atom limit and reduction step must be positive')
    atoms.calc.predictor.validate_atoms_data(atoms, atoms.calc.task_name)
    num_dof = 3 * len(atoms)
    batch_size = min(HESSIAN_BATCH_SIZE, max(1, HESSIAN_BATCH_MAX_ATOMS // len(atoms)))
    hessian = np.zeros((num_dof, num_dof), dtype=np.float64)
    start = 0
    while start < 2 * num_dof:
        count = min(batch_size, 2 * num_dof - start)
        try:
            forces = _displaced_forces(atoms, start, count, h, batch_size)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            failed_size = batch_size
            batch_size = max(1, batch_size - HESSIAN_BATCH_REDUCE_STEP)
            logging.warning('Hessian CUDA OOM for batch %d; reducing batch to %d',
                            failed_size, batch_size)
        else:
            for sample, force in enumerate(forces, start):
                hessian[:, sample // 2] += -force if sample % 2 == 0 else force
            start += count
            continue
        # Release the failed frame's tensors before clearing the CUDA cache.
        torch.cuda.empty_cache()

    hessian /= 2 * h
    hessian = 0.5 * (hessian + hessian.T)
    hessian *= EV2HARTREE / (ANG2BOHR ** 2)
    return hessian

@dataclass
class Job:
    job_id: str
    atomic_numbers: np.ndarray
    positions: np.ndarray
    charge: int
    multiplicity: int
    calc_type: str
    t_submit: float

@dataclass
class Result:
    job_id: str
    ok: bool
    data: bytes | None
    error: str | None


def _gpu_worker(
    in_q: mp.Queue,
    out_q: mp.Queue,
    model_path: str,
    device: str,
    hstep: float,
    inference_tf32: bool,
    inference_merge_mole: bool,
    inference_compile: bool,
    verbose: bool,
):
    """Keep one model resident and process requests in queue order."""
    # The parent handles Ctrl+C and sends the shutdown sentinel.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s GPU %(levelname)5s | %(message)s",
        force=True,
    )
    global MODEL_PATH, DEVICE, HESSIAN_STEP, INFERENCE_TF32, INFERENCE_MERGE_MOLE, INFERENCE_COMPILE
    MODEL_PATH, DEVICE, HESSIAN_STEP = model_path, device, hstep
    INFERENCE_TF32 = inference_tf32
    INFERENCE_MERGE_MOLE = inference_merge_mole
    INFERENCE_COMPILE = inference_compile
    logging.info(
        "GPU worker starting: MODEL=%s DEVICE=%s HESSIAN_STEP=%g tf32=%s merge_mole=%s compile=%s",
        MODEL_PATH, DEVICE, HESSIAN_STEP, INFERENCE_TF32, INFERENCE_MERGE_MOLE, INFERENCE_COMPILE,
    )

    calc = _build_calc()
    if INFERENCE_MERGE_MOLE or INFERENCE_COMPILE:
        logging.info("The first request initializes MoLE and compilation.")
    else:
        probe = molecule("H2")
        probe.info.update(charge=0, spin=1)
        probe.calc = calc
        probe.get_potential_energy()
        logging.info("Model warmup complete.")
    logging.info("GPU worker ready.")

    cached_atoms = None
    cached_atomic_numbers = None
    cached_charge = None
    cached_multiplicity = None

    while True:
        item = in_q.get()
        if item is None:
            break
        try:
            job = Job(**item)
            if (
                cached_atoms is not None
                and cached_charge == job.charge
                and cached_multiplicity == job.multiplicity
                and cached_atomic_numbers is not None
                and np.array_equal(cached_atomic_numbers, job.atomic_numbers)
            ):
                atoms = cached_atoms
                atoms.set_positions(job.positions)
            else:
                atoms = Atoms(numbers=job.atomic_numbers, positions=job.positions)
                atoms.calc = calc
                atoms.info["charge"] = job.charge
                atoms.info["spin"] = job.multiplicity
                atoms.set_pbc(False)
                cached_atoms = atoms
                cached_atomic_numbers = job.atomic_numbers.copy()
                cached_charge = job.charge
                cached_multiplicity = job.multiplicity

            if job.calc_type not in ("energy", "force", "hessian"):
                raise ValueError(f"unsupported calc_type: {job.calc_type}")

            t_comp0 = time.time()
            out = {"energy": _energy(atoms)}
            if job.calc_type in ("force", "hessian"):
                gradient = _gradient(atoms)
                out["gradient"] = gradient.reshape(-1).astype("<f8", copy=False)
            if job.calc_type == "hessian":
                hessian = _numeric_hessian(atoms, HESSIAN_STEP)
                out["hessian_lower"] = hessian[np.tril_indices(hessian.shape[0])]
            t_comp1 = time.time()

            if verbose:
                logging.info("job %s queue_wait=%.3fs compute=%.3fs", job.job_id, t_comp0 - job.t_submit, t_comp1 - t_comp0)

            out_q.put(Result(job.job_id, True, _raw_with_len(out), None))
        except Exception as e:
            logging.exception("GPU job failed")
            out_q.put(Result(item["job_id"], False, _raw_with_len(None, str(e)), str(e)))

    try:
        out_q.put_nowait(None)
    except pyqueue.Full:
        logging.warning("GPU output queue full while stopping; parent shutdown will close it.")

# Async server main process

class Server:
    def __init__(self, in_q: mp.Queue, out_q: mp.Queue, verbose: bool = False):
        self.in_q = in_q
        self.out_q = out_q
        self.verbose = verbose
        self.pending: dict[str, asyncio.Future] = {}
        self._next_job_id = 0
        self._drain_task: asyncio.Task | None = None
        self._server: asyncio.AbstractServer | None = None

    def _fail_pending(self, error: str):
        for job_id, fut in list(self.pending.items()):
            if not fut.done():
                fut.set_result(Result(job_id, False, None, error))
        self.pending.clear()

    async def start(self, host: str = "127.0.0.1", port: int = PORT, stop_event: asyncio.Event | None = None):
        backlog = 1024
        srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        srv_sock.bind((host, port))
        srv_sock.listen(backlog)
        srv_sock.setblocking(False)

        self._server = await asyncio.start_server(self.handle_client, sock=srv_sock, backlog=backlog)
        addrs = ", ".join(str(sock.getsockname()) for sock in self._server.sockets)
        logging.info("Unified UMA server listening on %s", addrs)

        self._drain_task = asyncio.create_task(self._drain_results())

        if stop_event is None:
            stop_event = asyncio.Event()
        try:
            await stop_event.wait()
        finally:
            await self.shutdown()

    async def shutdown(self):
        logging.info("Server shutdown start")
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

        if self._drain_task is not None:
            if not self._drain_task.done():
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.out_q.put, None)
                await self._drain_task

        self._fail_pending("server shutting down")
        logging.info("Server shutdown done")

    async def _drain_results(self):
        def _blocking_get():
            return self.out_q.get()

        while True:
            loop = asyncio.get_running_loop()
            res: Result | None = await loop.run_in_executor(None, _blocking_get)

            if res is None:
                self._fail_pending("gpu worker stopped")
                break
            fut = self.pending.pop(res.job_id, None)
            if fut is not None and not fut.done():
                fut.set_result(res)

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        t0 = time.time()
        if self.verbose:
            logging.info("Accepted %s", peer)
        job_id = None
        data = None
        t_recv = t_p0 = t_p1 = t_wait0 = t_wait1 = time.time()

        try:
            meta, payload = await _recv_job(reader)
            t_recv = time.time()
            payload_format = str(meta["payload_format"]).lower()
            if self.verbose:
                logging.info("[%s] Received job header+payload (format=%s payload_len=%d) in %.3f s",
                             peer, payload_format, len(payload), t_recv - t0)

            t_p0 = time.time()
            parsed = decode_wire_payload(payload, meta)
            t_p1 = time.time()
            if self.verbose:
                logging.info("[%s] Decode done in %.3f s (format=%s N=%d)",
                             peer, t_p1 - t_p0, payload_format, len(parsed.get("atomic_numbers", [])))

            job_id = str(self._next_job_id)
            self._next_job_id += 1
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            self.pending[job_id] = fut
            parsed["job_id"] = job_id
            parsed["t_submit"] = time.time()
            self.in_q.put_nowait(parsed)

            t_wait0 = time.time()
            res: Result = await asyncio.wait_for(fut, timeout=None)
            t_wait1 = time.time()
            data = res.data if res.data is not None else _raw_with_len(None, res.error or "worker failed")
        except pyqueue.Full:
            logging.warning("enqueue failed: queue full")
            if job_id is not None:
                self.pending.pop(job_id, None)
            data = _raw_with_len(None, "server busy")
        except asyncio.TimeoutError:
            if job_id is not None:
                self.pending.pop(job_id, None)
            t_wait1 = time.time()
            data = _raw_with_len(None, "timeout")
        except asyncio.CancelledError:
            if job_id is not None:
                self.pending.pop(job_id, None)
            raise
        except Exception as e:
            logging.exception("[%s] request failed", peer)
            if job_id is not None:
                self.pending.pop(job_id, None)
            data = _raw_with_len(None, str(e))
        finally:
            try:
                if data is not None and not writer.is_closing():
                    writer.write(data)
                    await writer.drain()
                    t_done = time.time()
                    if self.verbose:
                        logging.info("[%s] Sent %d bytes. stages: recv=%.3fs decode=%.3fs wait=%.3fs total=%.3fs",
                                     peer, len(data)-8, (t_recv - t0), (t_p1 - t_p0), (t_wait1 - t_wait0), (t_done - t0))
            except (ConnectionError, OSError):
                pass
            finally:
                try:
                    writer.close()
                    await writer.wait_closed()
                except (ConnectionError, OSError, RuntimeError):
                    pass


# Raw request batching

def queued_batches(in_q, max_size, max_atoms):
    """Collect ready jobs in FIFO order; Hessians and atom limits end a batch."""
    carry = None
    stopping = False
    while not stopping:
        first = carry if carry is not None else in_q.get()
        carry = None
        if first is None:
            return
        batch = [first]
        atoms = len(first['atomic_numbers'])
        if first['calc_type'] in ('energy', 'force'):
            while len(batch) < max_size and atoms < max_atoms:
                try:
                    item = in_q.get_nowait()
                except pyqueue.Empty:
                    break
                if item is None:
                    stopping = True
                    break
                size = len(item['atomic_numbers'])
                if item['calc_type'] not in ('energy', 'force') or atoms + size > max_atoms:
                    carry = item
                    break
                batch.append(item)
                atoms += size
        yield batch


class RawBatchEvaluator:
    def __init__(self, server, calc, *, merge_mole=False):
        from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

        self.server = server
        self.calc = calc
        self.collate = atomicdata_list_to_batch
        self.merge_mole = merge_mole
        self.merged_state = None

    def _atoms(self, job):
        if job.calc_type not in ('energy', 'force', 'hessian'):
            raise ValueError(f'unsupported calc_type: {job.calc_type}')
        if len(job.atomic_numbers) == 0 or not np.isfinite(job.positions).all():
            raise ValueError('Expected nonempty atoms with finite positions')
        atoms = Atoms(numbers=job.atomic_numbers, positions=job.positions, pbc=False)
        atoms.info.update(charge=job.charge, spin=job.multiplicity)
        self.calc.predictor.validate_atoms_data(atoms, self.calc.task_name)
        if self.merge_mole:
            state = (tuple(sorted(map(int, job.atomic_numbers))), job.charge, job.multiplicity)
            if self.merged_state is None:
                self.merged_state = state
            elif state != self.merged_state:
                raise ValueError('Merged MoLE requires a fixed composition, charge and spin. '
                                 'For heterogeneous jobs use --batch without --md and keep '
                                 'INFERENCE_MERGE_MOLE=False.')
        return atoms

    def _error(self, job_id, exc):
        message = str(exc)
        return self.server.Result(job_id, False,
                                  self.server._raw_with_len(None, message), message)

    def evaluate(self, items):
        """Yield one result per request, reporting input errors individually."""
        prepared = []
        for item in items:
            try:
                job = self.server.Job(**item)
                atoms = self._atoms(job)
                if job.calc_type == 'hessian':
                    atoms.calc = self.calc
                    out = {'energy': self.server._energy(atoms),
                           'gradient': self.server._gradient(atoms).reshape(-1)}
                    hess = self.server._numeric_hessian(atoms, self.server.HESSIAN_STEP)
                    out['hessian_lower'] = hess[np.tril_indices(len(hess))]
                    yield self.server.Result(job.job_id, True, self.server._raw_with_len(out), None)
                else:
                    prepared.append((job, self.calc.a2g(atoms)))
            except Exception as exc:
                logging.exception('Invalid/failed raw job %s', item['job_id'])
                yield self._error(item['job_id'], exc)
        if prepared:
            yield from self._predict(prepared)

    def _predict(self, prepared):
        """Split a batch on CUDA OOM; return other model errors to the clients."""
        import torch

        try:
            results = self._compute(prepared)
        except torch.cuda.OutOfMemoryError as exc:
            if len(prepared) == 1:
                yield self._error(prepared[0][0].job_id, exc)
                return
            logging.warning('CUDA OOM for %d jobs; retrying smaller batches', len(prepared))
        except Exception as exc:
            logging.exception('Raw batch prediction failed')
            for job, _ in prepared:
                yield self._error(job.job_id, exc)
            return
        else:
            yield from results
            return

        # Release the failed frame's tensors before clearing the CUDA cache.
        torch.cuda.empty_cache()
        middle = len(prepared) // 2
        yield from self._predict(prepared[:middle])
        yield from self._predict(prepared[middle:])

    def _compute(self, prepared):
        graphs = [graph for _, graph in prepared]
        data = graphs[0] if len(graphs) == 1 else self.collate(graphs)
        prediction = self.calc.predictor.predict(data)
        energies = prediction['energy'].detach().cpu().numpy().reshape(-1)
        forces = prediction['forces'].detach().cpu().numpy().reshape(-1, 3)
        sizes = [len(job.atomic_numbers) for job, _ in prepared]
        if len(energies) != len(prepared) or len(forces) != sum(sizes):
            raise ValueError('UMA returned inconsistent batch dimensions')
        if not np.isfinite(energies).all() or not np.isfinite(forces).all():
            raise ValueError('UMA returned non-finite energy/forces')
        results = []
        offset = 0
        for (job, _), energy, size in zip(prepared, energies, sizes):
            out = {'energy': float(energy) * self.server.EV2HARTREE}
            if job.calc_type == 'force':
                # Convert forces to gradients in Hartree/Bohr.
                grad = -forces[offset:offset + size] * self.server.EV2HARTREE / self.server.ANG2BOHR
                out['gradient'] = grad.reshape(-1).astype('<f8', copy=False)
            results.append(self.server.Result(job.job_id, True, self.server._raw_with_len(out), None))
            offset += size
        return results


def gpu_batch_worker(in_q, out_q, model_path, device, hstep, inference_tf32,
                     inference_merge_mole, inference_compile, verbose, max_size, max_atoms):
    server = sys.modules[__name__]

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s GPU %(levelname)5s | %(message)s', force=True)
    server.MODEL_PATH, server.DEVICE, server.HESSIAN_STEP = model_path, device, hstep
    server.INFERENCE_TF32 = inference_tf32
    server.INFERENCE_MERGE_MOLE = inference_merge_mole
    server.INFERENCE_COMPILE = inference_compile
    try:
        calc = server._build_calc()
        evaluate = RawBatchEvaluator(server, calc, merge_mole=inference_merge_mole)
        logging.info('Raw batching ready: max_size=%d max_atoms=%d tf32=%s merge_mole=%s compile=%s',
                     max_size, max_atoms, inference_tf32, inference_merge_mole, inference_compile)
        for items in queued_batches(in_q, max_size, max_atoms):
            start = time.perf_counter()
            for result in evaluate.evaluate(items):
                out_q.put(result)
            if verbose:
                logging.info('raw batch=%d atoms=%d compute=%.4fs job_ids=%s', len(items),
                             sum(len(x['atomic_numbers']) for x in items),
                             time.perf_counter()-start, [x['job_id'] for x in items])
    finally:
        try:
            out_q.put_nowait(None)
        except pyqueue.Full:
            logging.warning('GPU output queue full while stopping')


# i-PI client

def add_arguments(parser):
    group = parser.add_argument_group("i-PI force client (opt-in)")
    group.add_argument("--ipi", action="store_true", help="Connect to i-PI using shared MD settings instead of serving raw wrappers.")
    group.add_argument("--ipi-structure", help="ASE-readable template; atom order must match i-PI.")
    endpoint = group.add_mutually_exclusive_group()
    endpoint.add_argument("--ipi-unix", help="UNIX socket address, e.g. fastuma -> /tmp/ipi_fastuma; or absolute path.")
    endpoint.add_argument("--ipi-host", help="TCP host (default: localhost).")
    group.add_argument("--ipi-port", type=int, default=65535, help="TCP port (default: 65535).")
    group.add_argument("--ipi-charge", type=int, default=0)
    group.add_argument("--ipi-multiplicity", type=int, default=1)
    group.add_argument("--ipi-pbc", action="store_true", help="Use full 3D periodic boundaries and energy-derived stress; default is isolated.")
    group.add_argument("--ipi-connect-timeout", type=float, default=120.0, help="Seconds to wait for the server to start.")
    group.add_argument("--ipi-timeout", type=float, default=0.0, help="Socket I/O timeout in seconds; 0 waits indefinitely.")
    group.add_argument("--ipi-max-batch", type=int, default=32, help="Reject larger batches announced by i-PI before allocating them.")


def _message(name):
    return name.encode("ascii").ljust(12, b" ")


class IPIProtocolError(RuntimeError):
    """Invalid, truncated or incompatible message from the i-PI server."""


class IPIClient:
    """A persistent client serving one model.

    The evaluator receives (batch,3,3) cells and (batch,natoms,3) positions
    in i-PI's column-vector convention and atomic units, and returns energies,
    forces and wire-order virials in atomic units. It is called once per batch.
    """

    def __init__(self, sock, natoms, evaluate, *, max_batch=32, verbose=False):
        if natoms < 1 or max_batch < 1:
            raise ValueError("natoms and max_batch must be positive")
        self.sock = sock
        self.natoms = natoms
        self.evaluate = evaluate
        self.max_batch = max_batch
        self.verbose = verbose
        self.batch_size = 1
        self.state = "NEEDINIT"
        self.result = None
        self._buffers = {}

    def _read(self, size, *, allow_eof=False):
        # Reuse receive buffers. Evaluation completes before the next request.
        buf = self._buffers.get(size)
        if buf is None:
            buf = self._buffers[size] = bytearray(size)
        view = memoryview(buf)
        offset = 0
        while offset < size:
            count = self.sock.recv_into(view[offset:])
            if count == 0:
                if offset == 0 and allow_eof:
                    return None
                raise IPIProtocolError(f"Connection closed after {offset}/{size} bytes")
            offset += count
        return view

    def _int(self):
        return struct.unpack("=i", self._read(4))[0]

    def _array(self, shape):
        # Copy because subsequent fields may reuse a same-size receive buffer.
        count = int(np.prod(shape))
        return np.frombuffer(self._read(8 * count), dtype=np.float64).reshape(shape).copy()

    def _read_natoms(self):
        natoms = self._int()
        if natoms != self.natoms:
            raise IPIProtocolError(f"Atom count {natoms} != template {self.natoms}")
        return natoms

    def _initialize(self):
        if self.state == "HAVEDATA":
            raise IPIProtocolError("INIT received before pending forces were collected")
        self._int()  # replica id; all replicas use the same fixed template/state
        size = self._int()
        if not 0 <= size <= 65536:
            raise IPIProtocolError(f"Invalid INIT length: {size}")
        params = bytes(self._read(size)).decode("utf-8").strip(" \x00")
        batch = 1
        for item in params.split(","):
            if not item.strip():
                continue
            match = re.fullmatch(r"\s*batch_size\s*:\s*(\d+)\s*", item)
            if match is None:
                raise IPIProtocolError(
                    f"Unsupported INIT parameter: {item!r}; set charge/spin on the driver CLI"
                )
            batch = int(match[1])
        if not 1 <= batch <= self.max_batch:
            raise IPIProtocolError(f"i-PI batch_size={batch} exceeds allowed 1..{self.max_batch}")
        self.batch_size = batch
        self.state = "READY"

    def _read_positions(self):
        """Decode the two wire layouts before touching the model."""
        batch = self.batch_size
        if batch == 1:
            cell = self._array((1, 3, 3))
            inverse = self._array((1, 3, 3))
            natoms = self._read_natoms()
        else:
            # i-PI 3.3 batch protocol puts natoms before the cell blocks.
            natoms = self._read_natoms()
            cell_pairs = self._array((batch, 2, 3, 3))
            cell, inverse = cell_pairs[:, 0], cell_pairs[:, 1]
        positions = self._array((batch, natoms, 3))
        if not all(np.isfinite(a).all() for a in (cell, inverse, positions)):
            raise IPIProtocolError("Non-finite cell or positions")
        return cell, positions

    def _pack_result(self, outputs):
        batch = self.batch_size
        shapes = ((batch,), (batch, self.natoms, 3), (batch, 3, 3))
        arrays = []
        for value, shape in zip(outputs, shapes, strict=True):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid model output; expected finite {shape}, got {array.shape}")
            arrays.append(np.ascontiguousarray(array))
        energy, forces, virial = arrays
        # Consolidate the entire reply, including one empty JSON extras per bead.
        return b"".join((
            _message("FORCEREADY"), energy.tobytes(), struct.pack("=i", self.natoms),
            forces.tobytes(), virial.tobytes(), (struct.pack("=i", 2) + b"{}") * batch,
        ))

    def _positions(self):
        if self.state != "READY":
            raise IPIProtocolError(f"POSDATA received in state {self.state}")
        cell, positions = self._read_positions()
        started = time.perf_counter()
        self.result = self._pack_result(self.evaluate(cell, positions))
        self.state = "HAVEDATA"
        if self.verbose:
            logging.info("i-PI batch=%d atoms=%d compute=%.4fs", self.batch_size, self.natoms, time.perf_counter() - started)

    def run(self):
        while True:
            header = self._read(12, allow_eof=True)
            if header is None:
                if self.state == "HAVEDATA":
                    raise IPIProtocolError("Server disconnected with uncollected forces")
                return
            command = bytes(header).decode("ascii").strip()
            if command == "EXIT":
                return
            if command == "STATUS":
                self.sock.sendall(_message(self.state))
            elif command == "INIT":
                self._initialize()
            elif command == "POSDATA":
                self._positions()
            elif command == "GETFORCE":
                if self.state != "HAVEDATA":
                    raise IPIProtocolError(f"GETFORCE received in state {self.state}")
                self.sock.sendall(self.result)
                self.result = None
                self.state = "READY"
            else:
                raise IPIProtocolError(f"Unsupported i-PI command {command!r}")


class UMABatchEvaluator:
    """Evaluate independent structures together; forces require autograd."""

    def __init__(self, template, calc):
        from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

        self.template = template.copy()
        self.calc = calc
        self.collate = atomicdata_list_to_batch
        self.atoms = []
        calc.predictor.validate_atoms_data(self.template, calc.task_name)

    def __call__(self, cells, positions):
        count, natoms, _ = positions.shape
        if natoms != len(self.template):
            raise ValueError("Template atom count changed")
        if not self.atoms:
            self.atoms = [self.template.copy() for _ in range(count)]
        if len(self.atoms) != count:
            raise ValueError("Batch size changed during a run; restart the driver")
        cells_ase = cells.transpose(0, 2, 1) * units.Bohr
        pos_ase = positions * units.Bohr
        graphs = []
        for atoms, cell, pos in zip(self.atoms, cells_ase, pos_ase):
            if atoms.pbc.any() and np.linalg.det(cell) <= 0:
                raise ValueError("Periodic i-PI cell must have positive volume")
            atoms.set_cell(cell)
            atoms.set_positions(pos)
            graphs.append(self.calc.a2g(atoms))
        data = graphs[0] if count == 1 else self.collate(graphs)
        prediction = self.calc.predictor.predict(data)
        energy = prediction["energy"].detach().cpu().numpy().astype(np.float64).reshape(count)
        forces = prediction["forces"].detach().cpu().numpy().astype(np.float64).reshape(count, natoms, 3)
        if self.template.pbc.any():
            stress = prediction["stress"].detach().cpu().numpy().astype(np.float64).reshape(count, 3, 3)
            virial = -stress * np.linalg.det(cells_ase)[:, None, None]
        else:
            # Isolated-system virial: -dE/d(strain) = sum r (x) F.
            # Centering removes sensitivity to tiny residual net-force errors.
            centered = pos_ase - pos_ase.mean(axis=1, keepdims=True)
            virial = np.einsum("bni,bnj->bij", centered, forces)
        return energy / units.Ha, forces * (units.Bohr / units.Ha), virial.transpose(0, 2, 1) / units.Ha


def connect(args):
    if args.ipi_unix:
        family = socket.AF_UNIX
        endpoint = args.ipi_unix if args.ipi_unix.startswith("/") else "/tmp/ipi_" + args.ipi_unix
    else:
        family = socket.AF_INET
        endpoint = (args.ipi_host or "localhost", args.ipi_port)
    deadline = time.monotonic() + args.ipi_connect_timeout
    while True:
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            sock.settimeout(max(0.001, min(1.0, deadline - time.monotonic())))
            sock.connect(endpoint)
            if family == socket.AF_INET:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(args.ipi_timeout or None)
            logging.info("Connected to i-PI at %s", endpoint)
            return sock
        except OSError as exc:
            sock.close()
            if not isinstance(exc, TimeoutError) and exc.errno not in (errno.ENOENT, errno.ECONNREFUSED):
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Could not connect to i-PI at {endpoint}") from exc
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))


def run_ipi(args, build_calc):
    from ase.io import read

    if not args.ipi_structure:
        raise ValueError("--ipi requires --ipi-structure with the same atom order as i-PI")
    if args.ipi_multiplicity < 1 or not -100 <= args.ipi_charge <= 100:
        raise ValueError("Use positive spin multiplicity and charge within [-100, 100]")
    if args.ipi_max_batch < 1 or args.ipi_connect_timeout <= 0 or args.ipi_timeout < 0:
        raise ValueError("Invalid i-PI batch limit or timeout")
    if not 1 <= args.ipi_port <= 65535:
        raise ValueError("Invalid TCP port")
    atoms = read(args.ipi_structure, index=0)
    if not len(atoms) or not np.isfinite(atoms.positions).all():
        raise ValueError("Empty or non-finite template")
    # Ignore constraints and optional XYZ calculator results; i-PI owns dynamics.
    atoms.set_constraint()
    atoms.calc = None
    atoms.set_pbc(args.ipi_pbc)
    atoms.info = {"charge": args.ipi_charge, "spin": args.ipi_multiplicity}
    logging.info("i-PI UMA/omol: atoms=%d charge=%d multiplicity=%d pbc=%s", len(atoms), args.ipi_charge, args.ipi_multiplicity, args.ipi_pbc)
    if args.ipi_pbc:
        logging.info("Periodic virial uses energy-derived OMol stress (not fitted to stress labels).")
    calc = build_calc(periodic_stress=args.ipi_pbc)
    evaluate = UMABatchEvaluator(atoms, calc)
    # Initialize merged weights and compilation with the first actual batch.
    with connect(args) as sock:
        IPIClient(sock, len(atoms), evaluate, max_batch=args.ipi_max_batch, verbose=args.verbose).run()
    logging.info("i-PI driver finished.")


# Entrypoint

async def _watch_gpu_worker(process, server, stop_event):
    """Stop accepting work if the child exits, including during model loading."""
    while process.exitcode is None:
        await asyncio.sleep(0.2)
    logging.error("GPU worker exited (code=%s); stopping raw service", process.exitcode)
    server._fail_pending("gpu worker stopped")
    stop_event.set()


def main():
    parser = argparse.ArgumentParser(description="UMA-CBridge raw-wrapper server or i-PI force client")
    parser.add_argument(
        "--md",
        action="store_true",
        help="Use the MD_INFERENCE_* settings at the top of this script (also used by --ipi).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-request decode, queue, compute, and response timing logs.",
    )
    parser.add_argument('--batch', action='store_true',
                        help='Batch queued raw energy/gradient requests.')
    add_arguments(parser)
    args = parser.parse_args()
    if not MODEL_PATH or not os.path.isfile(MODEL_PATH):
        parser.error("set MODEL_PATH at the top of this script to an existing UMA checkpoint (.pt)")
    if args.batch and args.ipi:
        parser.error('--batch is for raw wrappers; i-PI batch size is set in its XML')
    if args.batch and (RAW_BATCH_MAX_SIZE < 1 or RAW_BATCH_MAX_ATOMS < 1):
        parser.error('RAW_BATCH_MAX_SIZE and RAW_BATCH_MAX_ATOMS must be positive')

    global INFERENCE_TF32, INFERENCE_MERGE_MOLE, INFERENCE_COMPILE
    if args.md or args.ipi:
        INFERENCE_TF32 = MD_INFERENCE_TF32
        INFERENCE_MERGE_MOLE = MD_INFERENCE_MERGE_MOLE
        INFERENCE_COMPILE = MD_INFERENCE_COMPILE

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)5s | %(message)s",
        force=True,
    )
    if args.ipi:
        logging.info(
            "i-PI shared MD settings: tf32=%s merge_mole=%s compile=%s MODEL=%s DEVICE=%s",
            INFERENCE_TF32, INFERENCE_MERGE_MOLE, INFERENCE_COMPILE, MODEL_PATH, DEVICE,
        )
        try:
            run_ipi(args, _build_calc)
        except KeyboardInterrupt:
            logging.info("i-PI driver stopped.")
        return
    _serve_raw(args)


def _serve_raw(args):
    """Run a model worker behind the raw socket server."""
    logging.info(
        "Unified UMA server (async + binary wire + gpu worker) PORT=%d MODEL_PATH=%s DEVICE=%s HESSIAN_STEP=%g md=%s verbose=%s tf32=%s merge_mole=%s compile=%s",
        PORT,
        MODEL_PATH,
        DEVICE,
        HESSIAN_STEP,
        args.md,
        args.verbose,
        INFERENCE_TF32,
        INFERENCE_MERGE_MOLE,
        INFERENCE_COMPILE,
    )

    ctx = mp.get_context("spawn")
    in_q: mp.Queue = ctx.Queue(maxsize=4096)
    out_q: mp.Queue = ctx.Queue(maxsize=4096)

    worker = _gpu_worker
    worker_args = dict(
        in_q=in_q,
        out_q=out_q,
        model_path=MODEL_PATH,
        device=DEVICE,
        hstep=HESSIAN_STEP,
        inference_tf32=INFERENCE_TF32,
        inference_merge_mole=INFERENCE_MERGE_MOLE,
        inference_compile=INFERENCE_COMPILE,
        verbose=args.verbose,
    )
    if args.batch:
        worker = gpu_batch_worker
        worker_args.update(max_size=RAW_BATCH_MAX_SIZE, max_atoms=RAW_BATCH_MAX_ATOMS)
    gpu_p = ctx.Process(target=worker, kwargs=worker_args, daemon=True)
    gpu_p.start()

    async def runner():
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, stop_event.set)
            loop.add_signal_handler(signal.SIGTERM, stop_event.set)
        except NotImplementedError:
            pass

        srv = Server(in_q, out_q, verbose=args.verbose)
        watcher = asyncio.create_task(_watch_gpu_worker(gpu_p, srv, stop_event))
        try:
            await srv.start("127.0.0.1", PORT, stop_event)
        finally:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher

    try:
        asyncio.run(runner())
        if gpu_p.exitcode not in (None, 0):
            raise RuntimeError(f"GPU worker failed with exit code {gpu_p.exitcode}")
    finally:
        try:
            in_q.put(None, timeout=1)
        except pyqueue.Full:
            logging.warning("GPU input queue is full during shutdown; terminating worker.")

        gpu_p.join(timeout=5)

        if gpu_p.is_alive():
            logging.warning("GPU worker did not exit after Ctrl+C; terminating it.")
            gpu_p.terminate()
            gpu_p.join(timeout=5)

        if gpu_p.is_alive():
            logging.warning("GPU worker still alive after terminate; killing it.")
            gpu_p.kill()
            gpu_p.join(timeout=2)

        in_q.close()
        in_q.cancel_join_thread()
        out_q.close()
        out_q.cancel_join_thread()


if __name__ == "__main__":
    main()
