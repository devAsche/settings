# 2026-11-16 RTX 3060 Ti → RTX 5060 Ti 16GB 換装に伴う AI 環境の最適化

対象PC: GALLERIA RM5C-R36T（i5-11400 / 32GB / ASUS PRIME B560M-A / 650W）
GPU: Palit RTX 5060 Ti 16GB Infinity 3、ドライバ 621.24

## 変更内容（`C:\AI model` 以下）

| ファイル | 変更 |
|---|---|
| `start_comfyui.bat` | `--lowvram` を削除し、`--use-sage-attention` を追加 |
| `launch_comfyui_task.ps1` | Museのタスク起動の引数に `--use-sage-attention` を追加（タスクの再登録で反映） |
| `autopilot\runner.py` | 1.4 → 1.5。`gpu_jobs.py` を読み込む数行を追加（他のジョブは変更なし） |
| `autopilot\gpu_jobs.py`（新規） | 読み取り専用ジョブ `gpu_check` |
| `autopilot\scripts\gpu_selftest.py`（新規） | ドライバ・PCIe・ReBAR・torch・Triton・torch.compile・SageAttention の自己診断 |
| `autopilot\config.json` | `pip_allowlist` に triton-windows と SageAttention のwheel URLを追加 |

venv（`C:\AI model\venv`, Python 3.12.14, torch 2.14.0+cu130）に追加したもの:

- `triton-windows==3.8.0.post29`
- `sageattention 2.2.0+cu130torch2.10.0andhigher.post6`（woct0rdho/SageAttention v2.2.0-windows.post6）

変更前のファイルは `before/`、変更後は `after/` にあります。PC上にも `*_backup.*` として残しています。

## 確認結果（`checks/`）

| 項目 | 変更前 | 変更後 |
|---|---|---|
| torch / CUDA / sm | 2.14.0+cu130 / 13.0 / sm_120 | 同左 |
| bf16 行列積 | 48.9 TFLOPS | 49.4 TFLOPS |
| Triton | なし | 3.8.0 OK |
| torch.compile | 失敗（Triton なし） | OK |
| SageAttention | なし | OK（SDPA比 約5.4倍、平均相対誤差 0.039） |

- PCIe はアイドル時 Gen2 x8（省電力時の挙動）、最大 Gen4 x8
- Resizable BAR はオフ（BAR1 = 256MiB）→ BIOS で有効化が必要

## 元に戻すとき

`before/` の各ファイルで上書きし、`pip uninstall sageattention triton-windows` を実行する。
