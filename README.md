# CUS Backend Deployment

Backend for receiving ultrasound studies from the Android handset, building 3D
volumes, and archiving images to one or more PACS nodes.

## Architecture

```
 probe / handset            backend (this repo)                 PACS node(s)
 ─────────────────          ──────────────────────              ──────────────
  uploads a study   ─────►  pacs-transfer-server.py   ─C-STORE─► Orthanc (primary)
  ONCE over WS/TCP          • saves to US_images/      ├───────► Backup PACS
  then can delete           • builds 3D volume (GPU)   └───────► Archive PACS
  its local copy            • fans out to ALL nodes
```

- The **handset uploads each study once** to the backend, then no longer needs
  to keep the images. It never talks to a PACS directly.
- The **backend** is what replicates to every PACS node (it is *not* PACS-to-PACS
  forwarding — the backend pushes to each node independently).
- `US_images/` on the backend is a working/staging area for reconstruction and
  retries. The long-term (28-year) archive lives on the **PACS nodes**.

### Services (docker-compose)

| Service        | Port            | Purpose                                  |
|----------------|-----------------|------------------------------------------|
| `backend`      | 8890 / 7556     | TCP + WebSocket image ingestion, 3D recon |
| `orthanc`      | 4242 / 8042     | Primary PACS (DICOM store + web UI)      |
| `ohif`         | 3000            | OHIF DICOM web viewer                    |
| `nifti-viewer` | 3001            | 3D volume browser for `outputs/`         |

## Deploy

```bash
./deploy.sh                # install Docker if needed, build, start everything
./deploy.sh status         # show running services + URLs
./deploy.sh logs backend   # follow backend logs
./deploy.sh stop           # stop all containers
./deploy.sh install-gpu    # install nvidia-container-toolkit (GPU recon)
```

## PACS nodes & redundancy (fallback archives)

Every study is C-STORE'd to the **primary PACS and every fallback node** you
configure. This gives the redundancy required for long-term (28-year) image
retention — if one archive is lost or offline, the others still hold the study.

**A study is only marked "sent" (`pacs_sent.flag`) once *every* node has
confirmed storage.** If a fallback node is offline, the flag is withheld and the
backend's janitor re-enqueues the study and retries the missing node(s) on a
later pass. Sends are idempotent (deterministic DICOM UIDs), so a retry to a
node that already has the study simply merges — nothing is duplicated.

### Primary PACS

Configured in [`docker-compose.yml`](docker-compose.yml) — this is the local
Orthanc container, reachable inside Docker as `orthanc`:

```yaml
- PACS_HOST=orthanc
- PACS_PORT=4242
- PACS_AET=ORTHANC
```

### Adding a second or third node

The easy path — no code edits. Copy the example env file, fill in each site, and
redeploy:

```bash
cp pacs-nodes.env.example .env
# edit .env:
#   PACS2_HOST=192.168.1.50
#   PACS2_PORT=104
#   PACS2_AET=BACKUP_PACS
./deploy.sh
```

You can add up to nine nodes total using the same numbered pattern
(`PACS2_*`, `PACS3_*`, … `PACS9_*`). For each node:

| Variable      | Meaning                          | Default if omitted |
|---------------|----------------------------------|--------------------|
| `PACSn_HOST`  | IP/hostname of the PACS          | *(node disabled)*  |
| `PACSn_PORT`  | DICOM port                       | `104`              |
| `PACSn_AET`   | Called AE title of the PACS      | `ORTHANC`          |

A node is only enabled when its `PACSn_HOST` is set, so leaving a block blank
simply skips it. The `.env` file is git-ignored so site addresses aren't
committed.

### Requirements for a fallback node to work

- **Reachability** — the IP must be reachable *from the backend container*.
  Outbound to a LAN/remote PACS works over the default bridge network; ensure
  the remote PACS firewall allows its DICOM port.
- **AE title whitelist** — the backend associates with calling AE title
  `PYNETDICOM`. If the remote PACS restricts which calling AE titles may store
  to it, add `PYNETDICOM` to its allow-list.

### Per-study override (optional)

The handset metadata may override the destinations for a single study:

- `pacs_nodes`: an explicit list `[{"ip", "port", "ae_title"}, …]` that
  replaces the env-configured nodes for that study.
- `pacs`: a single `{"ip", "port", "ae_title"}` dict merged into the primary
  node.
