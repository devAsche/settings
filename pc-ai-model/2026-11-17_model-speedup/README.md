# 2026-11-17 画像・動画モデルの高速化セットアップ

- ランナー拡張: `autopilot\gpu_jobs.py` に `hf_download`（HFから1ファイル取得＋SHA-256照合）と `comfy_info`（ComfyUIバージョンとノード仕様）を追加。`gpu_selftest.py` に PCIe 転送速度テストを追加（Gen4 x8、ピン留め 13.7GB/s を確認）。
- ダウンロード: Qwen-Image-2.1 Turbo int8 convrot / H3 ref2v Turbo 8step v1.1 LoRA / H3-Image pruned NVFP4（LSDJesus）。
- カスタムノード: ComfyUI-MiniMaxH3-CLIPCached（MIT, commit 507c1ec, 実行用ファイルのみ）。
- これまでの計測（同条件）: Qwen GGUF→int8 で 330.9s→80.5s、H3 ref2va int8→NVFP4 で 2,013s→1,785s。
- 計測手順は `setup_and_test_plan.md`（Muse への指示 014）。
