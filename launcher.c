// recipe single executable: [this ELF][zstd(tar of the PackageCompiler app)][trailer]. From the U0 spike launcher.
// The first run extracts into $RECIPE_CACHE (default $XDG_CACHE_HOME/recipe or ~/.cache/recipe)/<key>/; every run then
// execs <key>/bin/<app> with the original arguments, where julia_main parses the CLI (cli.rs).
// Pack mode, only without a trailer: launcher --pack <payload.tar.zst> <key> <app> <output>. Build: cc -O2 launcher.c -lzstd
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <ftw.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zstd.h>

#define MAGIC "RECIPEJL"
struct trailer { char magic[8]; uint64_t offset, size; char key[48], app[56]; };
_Static_assert(sizeof(struct trailer) == 128, "trailer layout");

static void die(const char *what) {
	fprintf(stderr, "recipe launcher: %s: %s\n", what, errno ? strerror(errno) : "invalid");
	exit(127);
}
static int write_all(int fd, const void *data, size_t n) {
	for (const char *p = data; n;) {
		ssize_t w = write(fd, p, n);
		if (w < 0) { if (errno == EINTR) continue; return -1; }
		p += w, n -= (size_t)w;
	}
	return 0;
}
static int mkdirs(char *path, mode_t mode) {
	for (char *p = path + 1; *p; p++)
		if (*p == '/') { *p = 0; if (mkdir(path, mode) && errno != EEXIST) return -1; *p = '/'; }
	return mkdir(path, mode) && errno != EEXIST ? -1 : 0;
}

// streaming tar reader: GNU and ustar, L/K long names, pax path/linkpath
struct tar {
	const char *root;
	unsigned char header[512];
	size_t have, collected;
	uint64_t remaining, padding;
	int fd, collect;
	char *buffer, longname[PATH_MAX], longlink[PATH_MAX];
};
static uint64_t octal(const unsigned char *f, size_t w) {
	uint64_t v = 0;
	if (f[0] & 0x80) { v = f[0] & 0x7f; for (size_t i = 1; i < w; i++) v = v << 8 | f[i]; return v; }
	for (size_t i = 0; i < w && f[i]; i++) if (f[i] >= '0' && f[i] <= '7') v = v * 8 + (uint64_t)(f[i] - '0');
	return v;
}
static int safe(const char *name) {
	if (name[0] == '/') return 0;
	for (const char *p = name; p; p = strchr(p, '/') ? strchr(p, '/') + 1 : NULL)
		if (p[0] == '.' && p[1] == '.' && (p[2] == '/' || !p[2])) return 0;
	return 1;
}
static void pax(struct tar *t, const char *r, size_t n) {
	for (const char *p = r, *end = r + n; p < end;) {
		char *sp;
		long len = strtol(p, &sp, 10);
		if (len <= 0 || p + len > end) return;
		const char *k = sp + 1, *le = p + len - 1, *eq = memchr(k, '=', (size_t)(le - k));
		if (eq) {
			size_t m = (size_t)(le - eq - 1), kl = (size_t)(eq - k);
			if (m >= PATH_MAX) m = PATH_MAX - 1;
			char *dst = kl == 4 && !memcmp(k, "path", 4) ? t->longname : kl == 8 && !memcmp(k, "linkpath", 8) ? t->longlink : NULL;
			if (dst) memcpy(dst, eq + 1, m), dst[m] = 0;
		}
		p += len;
	}
}
static void begin_entry(struct tar *t) {
	unsigned char *h = t->header;
	int i = 0;
	while (i < 512 && !h[i]) i++;
	if (i == 512) return;
	uint64_t size = octal(h + 124, 12);
	char type = (char)h[156], name[PATH_MAX], lnk[PATH_MAX], path[PATH_MAX], parent[PATH_MAX];
	t->remaining = size, t->padding = (512 - size % 512) % 512, t->fd = -1, t->collect = 0;
	if (type == 'L' || type == 'K' || type == 'x') {
		t->collect = type, t->collected = 0;
		if (!(t->buffer = malloc(size + 1))) die("memory");
		return;
	}
	if (type == 'g') return;
	if (t->longname[0]) snprintf(name, sizeof name, "%s", t->longname);
	else if (!memcmp(h + 257, "ustar\0", 6) && h[345]) snprintf(name, sizeof name, "%.155s/%.100s", (char *)h + 345, (char *)h);
	else snprintf(name, sizeof name, "%.100s", (char *)h);
	snprintf(lnk, sizeof lnk, t->longlink[0] ? "%s" : "%.100s", t->longlink[0] ? t->longlink : (char *)h + 157);
	t->longname[0] = t->longlink[0] = 0;
	if (!safe(name)) { errno = EINVAL; die(name); }
	snprintf(path, sizeof path, "%s/%s", t->root, name);
	for (size_t n = strlen(path); n > 1 && path[n - 1] == '/';) path[--n] = 0;
	mode_t mode = (mode_t)octal(h + 100, 8) & 07777;
	snprintf(parent, sizeof parent, "%s", path);
	char *slash = strrchr(parent, '/');
	if (slash && slash != parent) { *slash = 0; if (mkdirs(parent, 0755)) die(parent); }
	if (type == '5') { if (mkdir(path, mode | 0700) && errno != EEXIST) die(path); }
	else if (type == '2') { unlink(path); if (symlink(lnk, path)) die(path); }
	else if (type == '1') {
		char target[PATH_MAX];
		snprintf(target, sizeof target, "%s/%s", t->root, lnk);
		if (!safe(lnk) || link(target, path)) die(path);
	} else if (type == '0' || type == '\0' || type == '7') {
		if ((t->fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, mode | 0600)) < 0 || fchmod(t->fd, mode)) die(path);
	}
}
static void end_entry(struct tar *t) {
	if (t->fd >= 0) { if (close(t->fd)) die("close"); t->fd = -1; }
	if (!t->collect) return;
	t->buffer[t->collected] = 0;
	if (t->collect == 'L') snprintf(t->longname, sizeof t->longname, "%s", t->buffer);
	if (t->collect == 'K') snprintf(t->longlink, sizeof t->longlink, "%s", t->buffer);
	if (t->collect == 'x') pax(t, t->buffer, t->collected);
	free(t->buffer), t->buffer = NULL, t->collect = 0;
}
static void tar_feed(struct tar *t, const unsigned char *d, size_t n) {
	while (n) {
		size_t k;
		if (t->remaining) {
			k = n < t->remaining ? n : (size_t)t->remaining;
			if (t->fd >= 0 && write_all(t->fd, d, k)) die("write");
			if (t->collect) memcpy(t->buffer + t->collected, d, k), t->collected += k;
			d += k, n -= k, t->remaining -= k;
			if (!t->remaining && !t->padding) end_entry(t);
		} else if (t->padding) {
			k = n < t->padding ? n : (size_t)t->padding;
			d += k, n -= k, t->padding -= k;
			if (!t->padding) end_entry(t);
		} else {
			k = 512 - t->have < n ? 512 - t->have : n;
			memcpy(t->header + t->have, d, k), t->have += k, d += k, n -= k;
			if (t->have == 512) {
				t->have = 0, begin_entry(t);
				if (!t->remaining && !t->padding && !t->collect) end_entry(t);
			}
		}
	}
}
static void extract(int fd, const struct trailer *tr, const char *root) {
	ZSTD_DStream *s = ZSTD_createDStream();
	size_t ic = ZSTD_DStreamInSize(), oc = ZSTD_DStreamOutSize(), last = 1;
	unsigned char *in = malloc(ic), *out = malloc(oc);
	if (!s || !in || !out) die("memory");
	struct tar t = {.root = root, .fd = -1};
	off_t at = (off_t)tr->offset;
	for (uint64_t left = tr->size; left;) {
		ssize_t got = pread(fd, in, left < ic ? (size_t)left : ic, at);
		if (got <= 0) die("payload read");
		at += got, left -= (uint64_t)got;
		ZSTD_inBuffer input = {in, (size_t)got, 0};
		for (int full = 0; input.pos < input.size || full;) {  // a full output buffer means the decoder may hold more
			ZSTD_outBuffer output = {out, oc, 0};
			last = ZSTD_decompressStream(s, &output, &input);
			if (ZSTD_isError(last)) { fprintf(stderr, "recipe launcher: zstd: %s\n", ZSTD_getErrorName(last)); exit(127); }
			tar_feed(&t, out, output.pos), full = output.pos == oc;
		}
	}
	if (last) { errno = 0; die("payload truncated"); }
	ZSTD_freeDStream(s), free(in), free(out);
}
static int remove_entry(const char *p, const struct stat *s, int f, struct FTW *w) { (void)s, (void)f, (void)w; return remove(p); }
static int pack(int argc, char **argv, const char *self) {
	if (argc != 6) { fprintf(stderr, "usage: %s --pack <payload.tar.zst> <key> <app> <output>\n", argv[0]); return 2; }
	struct trailer tr = {0};
	memcpy(tr.magic, MAGIC, 8);
	if (strlen(argv[3]) >= sizeof tr.key || strlen(argv[4]) >= sizeof tr.app) { errno = 0; die("key or app name too long"); }
	strcpy(tr.key, argv[3]), strcpy(tr.app, argv[4]);
	int out = open(argv[5], O_WRONLY | O_CREAT | O_TRUNC, 0755);
	if (out < 0) die(argv[5]);
	char b[1 << 16];
	uint64_t written = 0;
	for (int round = 0; round < 2; round++) {
		int in = open(round ? argv[2] : self, O_RDONLY);
		if (in < 0) die(round ? argv[2] : self);
		if (round) tr.offset = written;
		for (ssize_t n; (n = read(in, b, sizeof b)) > 0;) {
			if (write_all(out, b, (size_t)n)) die("write");
			written += (uint64_t)n;
			if (round) tr.size += (uint64_t)n;
		}
		close(in);
	}
	if (write_all(out, &tr, sizeof tr) || close(out)) die("write");
	return 0;
}
int main(int argc, char **argv) {
	char self[PATH_MAX], root[PATH_MAX], dir[PATH_MAX], marker[PATH_MAX], exe[PATH_MAX], staging[PATH_MAX], done[PATH_MAX];
	ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
	if (n < 0) die("/proc/self/exe");
	self[n] = 0;
	int fd = open(self, O_RDONLY | O_CLOEXEC);
	struct stat st;
	struct trailer tr;
	if (fd < 0 || fstat(fd, &st)) die(self);
	if (!(st.st_size > 128 && pread(fd, &tr, 128, st.st_size - 128) == 128 && !memcmp(tr.magic, MAGIC, 8))) {
		close(fd);
		if (argc > 1 && !strcmp(argv[1], "--pack")) return pack(argc, argv, self);
		errno = 0;
		die("no payload; build with --pack");
	}
	tr.key[sizeof tr.key - 1] = tr.app[sizeof tr.app - 1] = 0;
	const char *cache = getenv("RECIPE_CACHE"), *xdg = getenv("XDG_CACHE_HOME"), *home = getenv("HOME");
	if (cache && *cache) snprintf(root, sizeof root, "%s", cache);
	else if (xdg && *xdg) snprintf(root, sizeof root, "%s/recipe", xdg);
	else if (home && *home) snprintf(root, sizeof root, "%s/.cache/recipe", home);
	else snprintf(root, sizeof root, "/tmp/recipe-%u", (unsigned)getuid());
	snprintf(dir, sizeof dir, "%s/%s", root, tr.key);
	snprintf(marker, sizeof marker, "%s/.complete", dir);
	snprintf(exe, sizeof exe, "%s/bin/%s", dir, tr.app);
	if (access(marker, F_OK)) {
		if (mkdirs(root, 0700)) die(root);
		snprintf(staging, sizeof staging, "%s/.%s.%d", root, tr.key, (int)getpid());
		if (mkdir(staging, 0755)) die(staging);
		extract(fd, &tr, staging);
		snprintf(done, sizeof done, "%s/.complete", staging);
		int m = open(done, O_WRONLY | O_CREAT, 0644);
		if (m < 0 || close(m)) die(done);
		if (rename(staging, dir)) {  // another process finished first, or a stale partial tree is in the way
			if (access(marker, F_OK)) { nftw(dir, remove_entry, 16, FTW_DEPTH | FTW_PHYS); if (rename(staging, dir)) die(dir); }
			else nftw(staging, remove_entry, 16, FTW_DEPTH | FTW_PHYS);
		}
	}
	close(fd);
	argv[0] = exe;
	execv(exe, argv);
	die(exe);
}
