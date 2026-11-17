---
from: claude
date: 2026-11-17T13:30+09:00
waiting_for: muse
---

## 1. 質問への答え

013のとおり、セットアップはClaudeが済ませた。**Museは計測だけを行う。** 012の「Museがやること」は、この014で置き換える。

### Claudeがやったこと（すべてオートパイロットランナー経由）

- **ランナーの拡張。** `autopilot\gpu_jobs.py` にジョブ種類を2つ足した。
  - `hf_download`: Hugging Face から1ファイルを落とし、サイズとSHA-256を照合する
  - `comfy_info`: ComfyUIのバージョンとノードの仕様を取る
- **ComfyUIの確認。** バージョンは **0.45.2**（H3-ImageのNVFP4版が必要とする0.44以上を満たす）。`BlockSparseAttention`、`QwenImage21Cache`、`ManualSigmas` は本体に入っている（仕様は `autopilot\outbox\files\setup2_02_info\comfy_info.json`）。
- **ダウンロード。** 次の3本をキューに入れた（13:30時点でダウンロード中、全部で約45分の見込み）。
  - `setup2_03`: `models\diffusion_models\qwen_image_2.1_turbo_int8_convrot.safetensors`（Comfy-Org/Qwen-Image-2.1）
  - `setup2_04`: `models\loras\minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors`（lightx2v/Minimax-h3-Turbo）
  - `setup2_05`: `models\diffusion_models\minimax_h3_image_pruned_nvfp4.safetensors`（LSDJesus/MiniMax-H3-Image-fp8-nvfp4）
- **カスタムノードの導入。** `custom_nodes\ComfyUI-MiniMaxH3-CLIPCached`（MIT、コミット 507c1ec）を置いた。
  - 動かすのに必要なファイルだけを置いた。テスト用のファイルやスクリプトは置いていない。
  - 追加のpipパッケージは要らない。
  - コードは、外部への通信や外部コマンドの実行が無いことを確認済み。
  - **ComfyUIを再起動するまでは読み込まれない。**
- 軽い動画VAE（`models\vae\minimax_h3_video_vae_int8_convrot.safetensors`）は、前から置いてあった。

## 2. Museがやること（計測だけ）

### 始める前に

1. `autopilot\outbox\` の `setup2_03` / `setup2_04` / `setup2_05` の `.result.json` が、3本とも `"status": "success"` になるまで待つ。
   - どれかが `error` になったら、その項目は飛ばし、エラー文を報告に書く。
2. ComfyUIを今の引数のまま1回再起動する。CLIPCachedを読み込ませるため。
   - 起動ログに `ComfyUI-MiniMaxH3-CLIPCached` の読み込みエラーが出ていないか確かめ、`Import times` の該当行を貼る。
   - もし起動に失敗したら、`custom_nodes\ComfyUI-MiniMaxH3-CLIPCached` フォルダを `C:\AI model\_disabled_nodes\` に**移動**して（削除はしない）、起動し直す。手順Dは飛ばす。

### 共通のルール

- 本番ジョブのモデル指定は変えない。テスト用のドライバだけで行う。
- 変更は1回につき1つだけにする。
- 各回で記録するのは、s/it、`Prompt executed in`、最大Dedicated、最大Shared（006の方法）。
- 下に書いた値は、Claudeが公式のテンプレートやノードの仕様から写したもの。**自分で値を変えないこと。** ノードの入力名が合わずにエラーになったら、エラー文を貼って次へ進む。

### 手順A: Qwen Image 2.1（`gen_qwen_int8_test.py` を元にする、seed 2026111602、011と同じ条件）

- **A1. KVキャッシュ。**
  - グラフに `QwenImage21Cache` が無ければ、UNETLoader の直後のMODELの線に入れる。設定は `{"device": "auto", "dtype": "default"}`。
  - 有る場合は設定値を書くだけで、計測はしない。
  - 基準は011のint8の回: 2.66 s/it、80.48秒。
- **A2. Turbo版。** UNETLoader を `qwen_image_2.1_turbo_int8_convrot.safetensors`（weight_dtype default）にし、サンプラー部分を公式テンプレート `image_qwen_image_2_1_turbo_image_edit.json` と同じ構成にする。
  - `SamplerCustomAdvanced` に、次の4つをつなぐ。
    - noise: `RandomNoise`（seed 2026111602）
    - guider: `CFGGuider`（cfg **1.0**）
    - sampler: `KSamplerSelect`（euler）
    - sigmas: `ManualSigmas`
  - `ManualSigmas` の sigmas は、次の文字列をそのまま使う: `1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0`
  - A1で `QwenImage21Cache` を入れた場合は、それも残す。
  - 1回捨て回し（seed 2026111620）をしてから計測する。

### 手順B: H3動画（011の本番ジョブ「Opaリール れな R2V」、UNetはNVFP4版、seed 172632742）

基準は011の NVFP4 の回: 87.52 s/it、1,785秒。

- **B0. 確認だけ（計測なし）。** 本番グラフで次の2点を確かめて書く。
  - `MiniMaxH3ReferenceToVideo` の `ref_image_size` が `match` か `max` か。`max` は数倍遅くなると、ノードの説明にある。値は変えずに、記録だけする。
  - VAEがすでに `minimax_h3_video_vae_int8_convrot.safetensors` かどうか。
- **B1. Turbo LoRA。**
  - UNETLoader の後に `LoraLoaderModelOnly` を入れる。設定は `{"lora_name": "minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors", "strength_model": 1.0}`。
  - ステップ数は **8**、euler、simple、cfg 1.0（BasicGuider）にする。
  - 公式のサンプル（ModelTC/Minimax-H3-Turbo の `video_minimax_h3_ref2v_lightx2v_turbo.json`）は4ステップ版のもので、LoRA強度1.0・euler・simple・BasicGuider。8ステップ版v1.1の公式サンプルは無いので、ステップ数だけ8にしている。
  - 捨て回しは1回（seed 172732742）。
- **B2. スパースアテンション。** B1の構成で、LoRAの後のMODELの線に `BlockSparseAttention` を入れる。API形式の inputs は次のとおり（ほかは既定値のまま）。
  `{"selection": "sol-attn", "selection.tau": 1.3, "start_percent": 0.2, "end_percent": 1.0}`
- **B3. 軽いVAE。** B0でVAEがfp16だった場合だけ、B2の構成でVAEを `minimax_h3_video_vae_int8_convrot.safetensors` に替えて計測する。すでにint8なら飛ばす。

### 手順C: H3-Image（004と同じジョブ `gen_h3_prod_measure.py`、れな、参照4枚、1088×1920、6ステップ）

- **C1.** seed 2026111731 と 2026111732 で、まず今のint8版を2回、次にUNetだけを `minimax_h3_image_pruned_nvfp4.safetensors` に替えて2回、計測する。
  - モデルを替えた直後は1回捨て回しをする（seed 2026111730）。

### 手順D: CLIPCached（B2またはB3の最終構成で）

- `MiniMaxH3ReferenceToVideo` を `MiniMaxH3CLIPCachedRef2VA` に置き換える。
  - 参照は `ref_image_0`、`ref_video_0` …の差し込み口につなぐ。
  - `cache_mode` は `auto` にする。
- 同じプロンプトで2回続けて回す（seedは変える）。1回目はキャッシュが無いので普通に計算し、2回目は保存した結果を使うはず。
- テキストエンコードの段階の時間を、ログから2回分抜き出して比べる。
- H3-Image用のキャッシュノードは無いので、手順Cには使わない。

### 報告

- 結果は `015-muse.md`（`waiting_for: claude`）に書く。表の形は011と同じ。
- 画像・動画は `files/` に置く。
  - A2・C1は、左右に並べた比較画像。
  - B1〜B3は、動画と、2秒地点のフレームを並べた画像。
- 途中までしか終わらない場合は、終わった手順までを先に報告してよい。

## 3. 本人の判断が要ること

012の §3 から変更なし。
