CC ?= cc
CPPFLAGS ?=
CFLAGS ?= -O3 -Wall -Wextra -std=c11
LDFLAGS ?=
LDLIBS ?=

WRAPPERS = wrappers/gaussian/gaussian_uma wrappers/orca/orca_uma wrappers/amber/orca wrappers/crest/orca
HEADER = wrappers/common/fastuma_protocol.h

.PHONY: all clean
all: $(WRAPPERS)

wrappers/gaussian/gaussian_uma: wrappers/gaussian/gaussian_uma.c $(HEADER)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

wrappers/orca/orca_uma: wrappers/orca/orca_uma.c $(HEADER)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

wrappers/amber/orca: wrappers/amber/orca.c $(HEADER)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

wrappers/crest/orca: wrappers/crest/orca.c $(HEADER)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f $(WRAPPERS)
