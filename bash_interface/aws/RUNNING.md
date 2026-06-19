# GNNPlus on AWS

Run GNNPlus paper experiments on AWS GPU instances (Docker). Mirrors `bash_interface/cluster/` for Harvard FASRC.

**Recommended instance:** `g5.4xlarge` (1× A10G, 64 GB RAM, ~$1.60/hr on-demand).  
Avoid `p5en.48xlarge` (8× H200) unless you parallelize 8 jobs — you pay for the whole node.

---

## Quickstart — what to do now

Assuming: AWS account ✓, budget ✓, IAM user + `aws configure --profile gnnplus` ✓

| Step | Where | Action |
|------|-------|--------|
| 1 | AWS Console | EC2 → Launch `g5.4xlarge`, Deep Learning GPU AMI, 200 GB disk, SSH key |
| 2 | Laptop | `aws configure --profile gnnplus` if not done |
| 3 | Laptop | `rsync` GNNPlus repo to the instance (see Part 4) |
| 4 | EC2 (SSH) | `sudo bash bash_interface/aws/setup_ec2_host.sh` |
| 5 | EC2 | `bash bash_interface/aws/build.sh` |
| 6 | EC2 | Export `WANDB_API_KEY`, run smoke test Docker command |
| 7 | EC2 | Run full paper job OR stop instance to save money |

**After training:** stop the instance (Step D in billing checklist).

---

## Part 1 — Create an AWS account

1. Go to [https://aws.amazon.com/](https://aws.amazon.com/) → **Create an AWS Account**.
2. Enter email, password, account name.
3. Choose **Personal** or **Professional** account.
4. Add a **credit/debit card** (required even for free tier; GPU instances are not free).
5. Verify phone number.
6. Select **Basic Support** (free).
7. Sign in to the [AWS Console](https://console.aws.amazon.com/).

### Secure the account (do once)

1. **IAM user** (don't use root for daily work):
   - Console → **IAM** → **Users** → **Create user**
   - User name e.g. `raphael-admin`
   - **Next** → **Attach policies directly** → check `AmazonEC2FullAccess`
   - **Create user**
   - Open user → **Security credentials** → **Create access key** (CLI use case)
   - Save Access Key ID + Secret (shown once)

2. **Install AWS CLI** on your laptop (use official v2 — avoids Homebrew Python/expat errors):
   ```bash
   # Option A — official installer (recommended on Mac)
   mkdir -p ~/aws-cli ~/.local/bin
   curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o /tmp/AWSCLIV2.pkg
   # See AWS docs for CurrentUserHomeDirectory install, or double-click the .pkg
   ln -sf ~/aws-cli/aws-cli/aws ~/.local/bin/aws
   export PATH="$HOME/.local/bin:$PATH"   # add to ~/.zshrc

   # Option B — Homebrew (if Option A fails)
   brew install awscli

   aws configure --profile gnnplus
   aws sts get-caller-identity --profile gnnplus
   ```

3. **Monthly budget with email alerts** (minimum — you did this ✓):
   - Console → **Billing** → **Budgets** → **Create budget**
   - Cost budget, monthly, e.g. $50–100
   - Alerts at 50%, 80%, 100% actual + 100% forecasted

### Billing safeguards checklist

AWS has **no true credit-card hard cap**. Alerts + discipline stop surprise bills.

| Step | Status | Action |
|------|--------|--------|
| Monthly budget + email alerts | ✓ done (you) | Billing → Budgets |
| **B. Auto-stop EC2 at budget limit** | ☐ TODO | Budget → **Actions** → Create action → **Stop EC2 instances** at 100% threshold |
| **C. Extra safeguards** | ☐ TODO | See below |
| **D. Stop instance when idle** | ☐ TODO | **Most important** — see below |

**B — Auto-stop EC2 when budget is hit (recommended):**
1. Billing → **Budgets** → open your budget
2. **Actions** → **Create action**
3. Action type: **Stop EC2 instances** (or Apply IAM policy to block new launches)
4. Threshold: **100%** of budgeted amount, notification type: **Actual**
5. Approval: **Automatic** → create IAM role when prompted

**C — Extra safeguards for GPU work:**
- Billing → **Billing preferences** → enable **Receive Billing Alerts**
- Enable **Receive Free Tier Usage Alerts**
- Tag instances: `Name=gnnplus-gpu`
- New accounts: **Service Quotas** → EC2 → request `G/VT` limit if g5 launch fails

**D — What actually limits your bill (GNNPlus):**
- GPU instances bill **per hour while running** (~$1.60/hr for g5.4xlarge)
- **Stop** when not training: EC2 → Instance state → **Stop instances**
  ```bash
  aws ec2 stop-instances --instance-ids i-xxxxxxxx --profile gnnplus
  ```
- Stopped: no compute charge; EBS disk still ~$16/mo for 200 GB
- **Terminate** when done if you don't need the volume

---

## Part 2 — Launch a GPU EC2 instance (detailed)

Use region **US East (N. Virginia) `us-east-1`** unless you have a reason not to (matches `aws configure`).

### 2.1 Verify CLI works (laptop)

```bash
export PATH="$HOME/.local/bin:$PATH"
aws sts get-caller-identity --profile gnnplus
```

Expected output (numbers will differ):

```json
{
    "UserId": "AIDA...",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/raphael-admin"
}
```

If this fails, finish `aws configure --profile gnnplus` first (Part 1).

---

### 2.2 Open EC2 in the console

1. Sign in: [https://console.aws.amazon.com/](https://console.aws.amazon.com/) (IAM user)
2. Top-right: confirm region is **US East (N. Virginia)**
3. Search bar → type **EC2** → open **EC2**
4. Left sidebar → **Instances** → orange button **Launch instances**

---

### 2.3 Name and tags

- **Name:** `gnnplus-gpu`

---

### 2.4 Application and OS Images (AMI)

1. Click **Browse more AMIs**
2. Search: `Deep Learning Base OSS Nvidia Driver GPU AMI`
3. Pick **Ubuntu 22.04** (64-bit x86) — published by **Amazon**
   - Full name looks like: *Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*
   - This AMI already has NVIDIA drivers; you still install Docker via our script
4. Click **Select**

If you can't find it: use **Ubuntu Server 22.04 LTS** instead — `setup_ec2_host.sh` installs everything.

---

### 2.5 Instance type

1. Click **Instance type**
2. Search: `g5.4xlarge`
3. Select **g5.4xlarge** (16 vCPU, 64 GiB RAM, 1× NVIDIA A10G 24GB)
4. **Do not** pick `p5en.48xlarge` (8× H200, ~$63/hr)

**Quota error?** New accounts often have 0 GPU quota. Fix:
- Console → search **Service Quotas**
- **Amazon Elastic Compute Cloud (Amazon EC2)** → **Running On-Demand G and VT instances**
- **Request increase at account level** → ask for **4** vCPUs → submit
- Approval can take hours to 2 days. Retry launch when approved.

---

### 2.6 Key pair (login)

1. **Key pair name** → **Create new key pair**
2. **Key pair name:** `gnnplus-key`
3. **Key pair type:** RSA
4. **Private key format:** `.pem` (Mac/Linux)
5. **Create key pair** → browser downloads `gnnplus-key.pem`
6. Move it somewhere safe, e.g.:
   ```bash
   mv ~/Downloads/gnnplus-key.pem ~/.ssh/gnnplus-key.pem
   chmod 400 ~/.ssh/gnnplus-key.pem
   ```

You need this file every time you SSH. If you lose it, you cannot SSH into the instance.

---

### 2.7 Network settings

1. Expand **Network settings**
2. **Firewall (security groups):** Create security group
3. **Security group name:** `gnnplus-ssh`
4. **Inbound rules:** ensure one rule exists:
   - **Type:** SSH
   - **Port:** 22
   - **Source:** **My IP** (not `0.0.0.0/0` — that exposes SSH to the whole internet)
5. Leave outbound as default (all traffic allowed)

---

### 2.8 Configure storage

1. Expand **Configure storage**
2. Change **Size (GiB)** from 8 → **200**
3. **Volume type:** gp3 (default is fine)

CIFAR10 datasets + Docker layers need space.

---

### 2.9 Launch

1. Right panel shows summary (~$1.62/hr for g5.4xlarge on-demand)
2. Click **Launch instance**
3. Click **View all instances**
4. Wait until **Instance state** = **Running** and **Status check** = **2/2 checks passed** (~2–5 min)

---

### 2.10 Get the public IP

On the **Instances** page, select `gnnplus-gpu`:

| Field | Example | Use |
|-------|---------|-----|
| **Public IPv4 address** | `3.85.123.45` | SSH and rsync |
| **Instance ID** | `i-0abc123...` | stop/start via CLI |

Copy the **Public IPv4 address** — call it `PUBLIC_IP` below.

**Note:** If you stop/start the instance, the public IP may change unless you attach an Elastic IP (optional).

---

### 2.11 SSH into the instance (laptop)

```bash
chmod 400 ~/.ssh/gnnplus-key.pem
ssh -i ~/.ssh/gnnplus-key.pem ubuntu@PUBLIC_IP
```

First connect may ask `Are you sure you want to continue connecting?` → type `yes`.

You should see a prompt like:

```
ubuntu@ip-172-31-xx-xx:~$
```

**Quick GPU check on the host:**

```bash
nvidia-smi
```

You should see an **NVIDIA A10G** and driver version. If `nvidia-smi` fails, wait 2 min (driver still loading) or you picked the wrong AMI.

---

### 2.12 Copy GNNPlus code from your Mac (laptop, new terminal tab)

Keep the SSH session open. On your **Mac** (not on EC2):

```bash
export PATH="$HOME/.local/bin:$PATH"

# Replace PUBLIC_IP with your instance IP
PUBLIC_IP=3.85.123.45

rsync -avz --progress \
  --exclude '.git' \
  --exclude 'results' \
  --exclude 'datasets' \
  --exclude 'wandb' \
  --exclude '__pycache__' \
  ~/Desktop/Academic_Research/Repos_GNN/GNNPlus/ \
  -e "ssh -i ~/.ssh/gnnplus-key.pem -o StrictHostKeyChecking=accept-new" \
  ubuntu@${PUBLIC_IP}:~/GNNPlus/
```

First rsync may take a few minutes. Re-run the same command after local edits to sync again.

---

### Option B — AWS CLI launch (advanced)

Only if you already created a key pair and security group in the console:

```bash
aws ec2 run-instances \
  --profile gnnplus \
  --region us-east-1 \
  --image-id resolve:ssm:/aws/service/deep-learning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --instance-type g5.4xlarge \
  --key-name gnnplus-key \
  --security-group-ids sg-XXXXXXXX \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=gnnplus-gpu}]'
```

---

## Part 3 — Prepare the host (one time, on EC2 via SSH)

SSH in first (`ssh -i ~/.ssh/gnnplus-key.pem ubuntu@PUBLIC_IP`), then run these **on the instance**:

### 3.1 Install Docker + NVIDIA container toolkit

```bash
cd ~/GNNPlus
sudo bash bash_interface/aws/setup_ec2_host.sh
```

Takes ~2–5 minutes. Ends with a note to run `nvidia-smi` in Docker.

### 3.2 Verify GPU works inside Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

You should see the A10G GPU listed. If this fails, re-run `setup_ec2_host.sh` or reboot the instance.

### 3.3 Create data directories (persists on EBS disk)

```bash
sudo mkdir -p /data/gnnplus/{datasets,results,wandb}
sudo chown -R "$USER:$USER" /data/gnnplus
ls -la /data/gnnplus/
```

Datasets, results, and W&B cache survive instance **stop/start** (same EBS volume). They are lost on **terminate**.

---

## Part 4 — Build and run with Docker

### 4.1 Build the Docker image (on EC2, ~10–15 min first time)

```bash
cd ~/GNNPlus
bash bash_interface/aws/build.sh
```

When done:

```bash
docker images | grep gnnplus
# gnnplus   gpu   ...
```

### 4.2 Set W&B credentials (on EC2)

Get your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize) (log in as `raphaelpellegrin`).

```bash
export WANDB_API_KEY="paste-your-key-here"
export WANDB_ENTITY="weber-geoml-harvard-university"
export WANDB_PROJECT="GNNPlus"
```

Optional — save in a file (don't commit to git):

```bash
cat > ~/.gnnplus_env <<'EOF'
export WANDB_API_KEY="paste-your-key-here"
export WANDB_ENTITY="weber-geoml-harvard-university"
export WANDB_PROJECT="GNNPlus"
EOF
chmod 600 ~/.gnnplus_env
source ~/.gnnplus_env
```

### 4.3 Smoke test (~5 epochs, ~10–20 min on GPU)

Use **tmux** so disconnecting SSH doesn't kill the job:

```bash
tmux new -s gnnplus
source ~/.gnnplus_env   # if you created it

docker run --gpus all --rm \
  -v /data/gnnplus:/data \
  -e WANDB_API_KEY \
  -e WANDB_ENTITY \
  -e WANDB_PROJECT \
  gnnplus:gpu \
  bash bash_interface/aws/smoke_test_cifar10_gatedgcn.sh
```

- Detach from tmux: `Ctrl-b` then `d`
- Reattach later: `tmux attach -t gnnplus`

Check W&B: [https://wandb.ai/weber-geoml-harvard-university/GNNPlus](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)  
Look for run `smoke_cifar10_gatedgcn_aws_ep5`.

### 4.4 Full CIFAR10 paper baselines (200 epochs × 3 models × 2 seeds)

```bash
tmux new -s cifar10
source ~/.gnnplus_env

docker run --gpus all --rm \
  -v /data/gnnplus:/data \
  -e WANDB_API_KEY -e WANDB_ENTITY -e WANDB_PROJECT \
  gnnplus:gpu \
  bash bash_interface/aws/cifar10_paper_baselines.sh
```

Runs all 6 jobs sequentially (~several hours on A10G). W&B run names: `cifar10_{gcn,gine,gatedgcn}_seed{0,1}_aws`.

### 4.5 Stop the instance when done (save money!)

**From AWS Console:** EC2 → Instances → select `gnnplus-gpu` → **Instance state** → **Stop instance**

**From your Mac:**

```bash
# Find instance ID in EC2 console, then:
aws ec2 stop-instances --instance-ids i-0abc123def456 --profile gnnplus --region us-east-1
```

Stopped = no GPU hourly charge. EBS (~200 GB) still costs ~$16/month.

**To start again later:** **Start instance** → get new **Public IP** → SSH → resume training (datasets in `/data/gnnplus` are still there).

---

### Sync code from Mac (repeat after local edits)

```bash
PUBLIC_IP=<new-ip-after-start>
rsync -avz --progress \
  --exclude '.git' --exclude 'results' --exclude 'datasets' \
  ~/Desktop/Academic_Research/Repos_GNN/GNNPlus/ \
  -e "ssh -i ~/.ssh/gnnplus-key.pem" \
  ubuntu@${PUBLIC_IP}:~/GNNPlus/
```

### Any paper dataset

```bash
docker run --gpus all --rm \
  -v /data/gnnplus:/data \
  -e WANDB_API_KEY -e WANDB_ENTITY -e WANDB_PROJECT \
  gnnplus:gpu \
  bash bash_interface/aws/run_paper.sh cluster 2
```

Same configs as `configs/{gcn,gine,gatedgcn}/<dataset>.yaml` — paper defaults, no overrides.

---

## Part 5 — Run without Docker (optional)

If you prefer conda on the host (same as cluster):

```bash
# Adapt cluster script paths for /data
export GNNPLUS_PROJECT_ROOT=~/GNNPlus
export GNNPLUS_DATASET_DIR=/data/gnnplus/datasets
export GNNPLUS_RESULTS_DIR=/data/gnnplus/results
export WANDB_API_KEY=...

# Edit create_gnnplus_env.sh paths or run pip install manually (see Dockerfile)
bash bash_interface/aws/smoke_test_cifar10_gatedgcn.sh
```

Docker is recommended — reproducible CUDA/PyG stack.

---

## Cost estimates (us-east-1, on-demand)

| Instance | GPU | $/hr | CIFAR10 paper (6 runs, ~20–40 GPU-hr) |
|----------|-----|------|----------------------------------------|
| g5.4xlarge | 1× A10G | ~$1.62 | ~$30–65 |
| g5.4xlarge Spot | 1× A10G | ~$0.5–0.9 | ~$10–35 |
| p5.4xlarge | 1× H100 | ~$6.88 | ~$140–275 |
| p5en.48xlarge | 8× H200 | ~$63 | **~$1,300+** (overkill) |

**Tips to save money:**
- Use **Spot instances** for batch training (can be interrupted).
- **Stop** the instance when not training (EBS storage still billed, ~$16/mo for 200 GB).
- **Terminate** when done if you don't need the data on that volume.

---

## File layout

```
bash_interface/aws/
  Dockerfile                      # GPU image (CUDA 12.1 + PyG cu121)
  build.sh                        # docker build
  setup_ec2_host.sh               # Docker + nvidia-container-toolkit
  common_env.sh                   # W&B, paths, import check
  entrypoint.sh                   # Docker entrypoint
  smoke_test_cifar10_gatedgcn.sh  # 5-epoch sanity check
  cifar10_paper_baselines.sh      # paper CIFAR10 (6 tasks)
  run_paper.sh                    # any dataset, 3 models
  RUNNING.md                      # this file
```

Persistent on EBS (`/data/gnnplus`):

| Path | Contents |
|------|----------|
| `datasets/` | Downloaded PyG datasets (reused across runs) |
| `results/` | Checkpoints, logs |
| `wandb/` | W&B offline cache |

---

## Harvard cluster vs AWS

| | Harvard FASRC | AWS |
|--|---------------|-----|
| Submit jobs | `sbatch bash_interface/cluster/...` | `docker run ... bash_interface/aws/...` |
| GPU | `mweber_gpu` partition | `g5.4xlarge` |
| Data | `/n/netscratch/...` | `/data/gnnplus` EBS |
| W&B entity | `weber-geoml-harvard-university` | same |
| Configs | `configs/` | same — no code changes |

---

## Troubleshooting

- **`docker: Error response from daemon: could not select device driver`**
  Run `sudo bash bash_interface/aws/setup_ec2_host.sh` and restart Docker.

- **`CUDA available: False` inside container**
  Use `--gpus all` and verify `nvidia-smi` on host.

- **Quota error launching g5**
  Console → **Service Quotas** → EC2 → request limit increase for `G/VT` instances (often 0 for new accounts).

- **W&B 401**
  Export `WANDB_API_KEY` before `docker run`.

- **`aws: Symbol not found: _XML_SetAllocTrackerActivationThreshold` (Mac)**
  Homebrew `awscli` conflicts with system libexpat. Install official CLI v2:
  ```bash
  bash bash_interface/aws/install_aws_cli_mac.sh
  source ~/.zshrc
  aws --version   # should work (exe/arm64)
  ```
  Ensures `~/.local/bin/aws` is before `/usr/local/bin/aws` in PATH.
