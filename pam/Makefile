PAM_MODULE = pam_faceauth.so
SRC = pam_faceauth.c
CFLAGS ?= -O2 -fPIC -Wall -Wextra
LDFLAGS ?= -shared
LDLIBS = $(shell pkg-config --libs dbus-1) -lpam
CPPFLAGS = $(shell pkg-config --cflags dbus-1)

all: $(PAM_MODULE)

$(PAM_MODULE): $(SRC)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -f $(PAM_MODULE)
