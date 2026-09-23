# Containers, explained

hvr-cnn ships as a container image: a frozen Linux system with Python,
PyTorch, the MINC tools, the network weights and the template inside. This
page explains how it is run and why every option is there, so the commands
stop being magic. `bin/hvr-cnn-container` writes these commands for you;
`hvr-cnn-container --print ...` shows exactly what it would run.

## 1. The idea in four sentences

An *image* is a read-only file system plus a default command (here:
`hvr-cnn`). A *container* is one run of that image: a process that sees the
image's files instead of your machine's, and **none of your files unless you
mount them**. When the process ends, the container is gone (`--rm`);
anything it should keep must be written to a mounted directory. Podman,
Docker and Apptainer are three programs that do this, with different
defaults.

## 2. Start from what the command needs

The container options follow from what the hvr-cnn command reads and
writes on *your* disk:

| command | reads from your disk | writes to your disk | so the container needs |
|---|---|---|---|
| `selftest` | nothing (checks packages, programs, weights and templates *inside the image*, runs each model on a blank patch, writes and deletes one test file in the container's `/tmp`) | nothing | nothing mounted; any user |
| `run` | the scans (`-i`, or the `--csv` table and the scans it lists) | the output folder `-o` (and `--work-dir` if given) | inputs mounted read-only, the output folder mounted writable, running as you |
| `check` | the output folder and `--reference` | nothing | those folders mounted read-only |

So `podman run --rm --read-only --network none IMAGE selftest` is complete:
no mounts, no user options. `run` is the command that needs all of them.

## 3. Podman, option by option

```sh
podman run --rm --read-only --network none \
    --userns=keep-id --user "$(id -u):$(id -g)" \
    --volume /data/study:/data/study:ro \
    --volume /data/results:/data/results \
    --workdir "$PWD" \
    ghcr.io/soffiafdz/hvr-cnn:0.1.0 \
    run -i /data/study/sub-01_T1w.nii.gz -o /data/results
```

| option | what it does | why we use it |
|---|---|---|
| `run` | start a container from an image | |
| `--rm` | delete the container when it exits | otherwise every run leaves a stopped container behind (`podman ps -a`) |
| `--read-only` | the image's file system cannot be written | proves hvr-cnn writes nothing inside the image; podman still gives writable, in-memory `/tmp`, `/var/tmp`, `/run` |
| `--network none` | no network inside | hvr-cnn never downloads anything; this guarantees it |
| `--userns=keep-id` | rootless podman maps users into a private range; `keep-id` maps *your* user id to the same number inside | without it, files written to your folders would belong to an odd high user id |
| `--user UID:GID` | run the process as that user | **the image's default user wins over `keep-id` alone**: without this, the process ran as the image's user and could not write to your folder (see 6.2) |
| `--volume HOST:CONTAINER[:ro]` | mount a host directory inside; `:ro` = read-only | the only way the container sees your files. We mount each directory **at the same path**, so `/data/x.mnc` means the same thing inside and out and the command line needs no translation |
| `--workdir DIR` | the process's current directory | relative paths on the command line (`-o results`) resolve against it; we set it to your current directory (and mount it read-only) |
| `IMAGE` | which image | a registry reference; a version tag (`:0.1.0`) makes runs reproducible, `:latest` moves |
| everything after the image | the command | passed to the image's entry point, `hvr-cnn`: `run ...`, `selftest`, `check ...` |

**Docker** is the same except: no `--userns=keep-id` (use `--user` alone),
and `/tmp` must be made writable explicitly with `--tmpfs /tmp`.

On a Mac with Apple Silicon add `--platform linux/amd64`: the image is built
for Intel/AMD processors and runs under emulation (slower).

## 4. Apptainer / Singularity (clusters)

```sh
apptainer run --containall --cleanenv --no-home \
    --bind /scratch/me/tmp1:/tmp --bind /scratch/me/tmp2:/var/tmp \
    --bind /data/study:/data/study:ro --bind /data/results:/data/results \
    --pwd "$PWD" hvr-cnn_0.1.0.sif run -i /data/study/sub-01_T1w.nii.gz -o /data/results
```

Apptainer's defaults are the opposite of podman's: it runs as you, sees the
network, and mounts your home, the current directory and `/tmp`
automatically. We switch that off for reproducibility:

| option | what it does | why |
|---|---|---|
| `--containall` | no automatic mounts of home, `/tmp`, current directory | the run depends only on what you mount explicitly |
| `--cleanenv` | do not pass your environment variables in | a stray `PYTHONPATH` or `OMP_NUM_THREADS` on the host cannot change the run |
| `--no-home` | do not mount your home | same |
| `--bind DIR:/tmp`, `--bind DIR:/var/tmp` | real disk space for temporary files | with `--containall`, `/tmp` and `/var/tmp` become small in-memory file systems (the site's "sessiondir max size": 16 MB at BIC); hvr-cnn's intermediate files and N3 overflow it (see 6.3) |
| `--bind HOST:CONTAINER[:ro]` | like podman's `--volume` | |
| `--pwd DIR` | like `--workdir` | |
| `X.sif` or `docker://REF` | a SIF file, or an image pulled and converted on the fly | build the SIF once (`apptainer build hvr-cnn.sif docker://REF`), then reuse it |

## 5. Building the image (maintainers)

```sh
podman build --platform linux/amd64 -f container/Dockerfile \
    --build-arg VERSION=0.1.0 --build-arg REVISION=$(git rev-parse HEAD) \
    -t hvr-cnn:0.1.0 .
```

`-f` names the recipe, `--build-arg` fills the `ARG`s used in the image's
labels, `-t` names the result, and `.` is the *build context*: the files
the recipe may copy. `.dockerignore` is an allow-list, so test data under
`tests/<subject>/` can never enter the image.

## 6. What went wrong on the way, and why the fix works

These are real failures from setting this up; each teaches one rule.

### 6.1 `zsh: no such file or directory: podman run --rm ...`

*What happened:* the command was stored in a variable,
`H="podman run --rm ..."`, and run as `$H run ...`. Bash splits an unquoted
variable into words; **zsh does not**, so zsh looked for a single program
literally named `podman run --rm ...`.
*Fix:* a shell function, `H() { podman run --rm ... "$@"; }`. A function's
body is parsed as a command line in both shells, and `"$@"` passes the
arguments through unchanged. *Rule:* never store commands in strings.

### 6.2 `PermissionError: [Errno 13] Permission denied: '/kit/mine'`

*What happened:* with `--userns=keep-id` alone, podman kept the user
declared in the image (the base image's `mambauser`), which is not you and
cannot write to your directory.
*Fix:* add `--user "$(id -u):$(id -g)"`. `keep-id` makes your user id exist
inside the container with the same number; `--user` makes the process
actually run as it. *Rule:* in rootless podman use both.

### 6.3 `HDF5 ... No space left on device` in the second N3 pass (MNI pipeline image)

*What happened:* the disk was not full. Under Apptainer `--containall`,
`/var/tmp` is a 16 MB in-memory file system, and N3 writes its work files
to `/var/tmp`. The first diagnosis ("the host's /tmp is full") was wrong:
`df` showed /tmp 10 % used; running `df -h /tmp /var/tmp` *inside* the
container showed the 16 MB `/var/tmp`.
*Fix:* bind real disk space to `/var/tmp` as well as `/tmp`. hvr-cnn also
passes N3 an explicit scratch directory inside its own work directory, so
it no longer depends on `/var/tmp` at all. *Rule:* when a tool reports a
full disk, check the file system **inside** the container.

### 6.4 The run starts, then nothing happens (MNI pipeline image)

*What happened:* the pipeline's tasks each ask its scheduler (Ray) for
`--threads` CPUs from a pool of `--prl` CPUs. With `--prl 1 --threads 4`
no task can ever get 4 CPUs, so the run waits forever without an error.
*Fix:* `--prl` >= `--threads` (the `mni-lng-pipeline` wrapper sets them
equal). *Rule:* a silent hang with 0 % CPU is a scheduling deadlock, not
slowness; `ps` shows it.

### 6.5 `hvr-cnn-container -o .` mounted the same folder twice

*What happened:* the output `.` became `/path/.` while the working directory
was `/path`: the same folder, two different strings, mounted once
read-write and once read-only, so the output could end up read-only.
*Fix:* normalise every path (`.`, `..`, `//`) before comparing, so one
directory is always one string. *Rule:* compare normalised paths.

### 6.6 `Error: statfs /home/.../input: no such file or directory`

*What happened:* `input/scan.mnc` was typed in a different folder than the
one holding `input/`, so the relative path pointed nowhere. `statfs` is the
system call podman uses to look at a folder before mounting it; it was
asked to mount a folder that does not exist.
*Fix:* the wrapper now checks every input first and names the folder
relative paths were resolved against, and creates output folders only after
that. *Rule:* relative paths are relative to where you are (`pwd`), not to
where the files are.

## 7. Exercise: build the podman command one piece at a time

Run these in order (on a Linux machine with podman and the image). Steps 4
and 5 are meant to fail: reading those errors is the point.

0. `export IMG=localhost/hvr-cnn:dev` (a locally built image is called
   `localhost/<name>:<tag>`).
1. **Minimum:** `podman run $IMG selftest`, then `podman ps -a`: it
   worked, but the finished container is still listed. `podman rm -a`
   cleans up.
2. **`--rm`:** `podman run --rm $IMG selftest`; `podman ps -a` is now
   empty.
3. **Lock down:** `podman run --rm --read-only --network none $IMG
   selftest` still passes: hvr-cnn writes nothing into the image and needs
   no network.
4. **Files, unmounted:** in a folder with a scan,
   `podman run --rm --read-only --network none $IMG run -i input/scan.mnc -o out`
   fails with "no such file": the container sees only the image.
5. **Mount it:** add `--volume $PWD:$PWD --workdir $PWD`. Now it fails
   with "no permission": the process runs as the image's user (6.2).
6. **Run as yourself:** add `--userns=keep-id --user $(id -u):$(id -g)`.
   It works, and `ls -l out` shows the files are yours.
7. **Compare:** `hvr-cnn-container --print run -i input/scan.mnc -o out`.
   The only difference: the wrapper mounts the output read-write and the
   current folder read-only (`:ro`), so inputs cannot be modified by
   accident.
