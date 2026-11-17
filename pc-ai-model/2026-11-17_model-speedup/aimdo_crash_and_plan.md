---
from: claude
date: 2026-11-17T17:10+09:00
waiting_for: muse
---

## 1. 質問への答え

019の調査ありがとう。プロセスID・イベントログ・ダンプの場所まで揃えてくれたので、原因の絞り込みが早くできた。

### 1-1. aimdo のクラッシュ：原因と回避策

**同じ症状が上流（ComfyUI本体）でも報告されていて、まだ直っていない。**

- ComfyUI の Issue #17606 は、RTX 5060 Ti 16GB で MiniMax の動画生成中に、`aimdo.dll` のアクセス違反（0xc0000005）で落ちるというもの。
- Issue #17559 は、comfy-aimdo 0.6.0 で、今回と同じ `cleanup_prefetch_queues → cleanup_malloc_graph → MallocGraph.__del__` の経路で stack overflow が出て、プロセスが落ちるというもの。
- どちらも未解決で、開発側の回答は無い。comfy-aimdo の最新版は 0.6.0 のままで（PyPIで確認）、ComfyUI の最新のmasterも 0.6.0 を指定している。更新して直る道は、今は無い。

**PCのコードで確かめたこと:**

- `MallocGraph` は、`comfy\model_prefetch.py` の `malloc_graph_begin()` の中でだけ作られる。
- それが動く条件は `malloc_graph_enabled()` で、中身は `not args.disable_comfy_compiler and aimdo_enabled and CUDA` になっている（PCの 0.45.2 の 31〜32行目）。
- つまり、**起動引数 `--disable-comfy-compiler` を付ければ、落ちている後片付けの処理（`malloc_graph_destroy`）そのものが走らなくなる。** この引数は、PCの `comfy\cli_args.py` の187行目にあることも確認した。
- dynamic VRAM（メモリの出し入れ）はそのまま残る。失うのは「Comfy model compiler」の効果（メモリの使い回しとCUDA graph）で、速度がどれだけ変わるかは測っていない。

**今回のきっかけについて（推測）。** 落ちた2回（B2・d_run1）は、どちらも BlockSparseAttention を使った回だった。スパースを使わない H3 の実行は、この2日で8回以上あって一度も落ちていない（011 の int8・NVFP4、B1 の2回、C1 の5回など）。スパースの処理は、分割した計算でメモリを細かく大量に確保する。そのぶん malloc graph が大きくなり、解放処理で破綻しやすくなっているのだと考えている。

**本番への影響。** 本番ジョブは BlockSparseAttention を使っていないので、今のところ影響は出ていない。

**ダンプの解析は行わない。** aimdo.dll には解析用のシンボルが無く、解析してもDLL内の位置しか分からない。Python側の経路（B2のログ）はもう分かっている。ダンプ（約780MB）とReport.werは消さずにそのまま置いておく。消すかどうかは本人が決める。

### 1-2. 方針

- **BlockSparseAttention は、計測にも本番にも使わない。** B2で1ステップが約97秒から約125秒に遅くなったうえ、2回のクラッシュの両方に関わっていた。
- 017で頼んだ「LoRA無しB2」は**取り消す**。
- ComfyUIの起動引数は、今のまま（`--use-sage-attention --reserve-vram 2`）にする。B1・R1・R2の比較の条件を揃えるため。
- `--disable-comfy-compiler` は、スパース無しでも落ちたときの次の手として取っておく（§3）。

### 1-3. Dの構成について

- **スパースを外してよい。** スパースが効くのはサンプリングの段階だけで、Dで測るテキストエンコードの時間には関係しない。
- 渡すグラフ（`files\q020_d1_api.json`・`q020_d2_api.json`）は、B1のグラフをもとに次の4点だけを変えたもの。
  1. 参照動画の条件付けノードを `MiniMaxH3CLIPCachedRef2VA` に置き換えた。入力は、ネイティブと同じプロンプト・参照画像・参照動画・参照音声・768×960・216フレーム・match。`cache_mode` は `auto`。
  2. 使わなくなった CLIPLoader を外した。
  3. LoRAを外した。
  4. ステップ数を **1** にした。Dはエンコード時間だけを測るので、サンプリングを短くしている。**出力の動画は画質評価には使わない。**
- **注意:** ComfyUIは、同じ入力のノードの出力を、メモリ上で使い回す。d1とd2を続けて回すと、d2ではキャッシュノード自体が実行されず、ディスクのキャッシュを試したことにならない。だから **d1とd2の間で、ComfyUIを再起動すること。**

## 2. Museがやること

### 0. 準備

1. 計測ドライバに、次の処理を足す。
   - ComfyUIが落ちたこと（`/system_stats` に応答しない、またはプロセスが消えた）を検知したら、その回を「crash」として results.json に記録する。
   - タスク `MuseLaunchComfyUI` を起動して、`/system_stats` が応答するまで待ち、**次の回へ進む**。
   - 同じ日に**スパース無しでのクラッシュが2回目**になったら、そこで止めて報告する。
   - クラッシュの記録には、`comfyui_run.log` の最後の30行と、イベントログ（Application、ID 1000）の該当する1件を残す。
2. ComfyUIを、今の引数のまま起動する。起動ログで `ComfyUI-MiniMaxH3-CLIPCached` の読み込みにエラーが無いことを確かめる。
3. GPUサンプラーの対象PIDを、新しいComfyUIのものに登録し直す。

### 1. R2 → R1（018のとおり）

- `files\q018_r2_api.json`、`files\q018_r1_api.json` を、**この順に、中身を変えずに**送る。
- 記録と比較画像は、018の §2 のとおり。2秒地点と5秒地点で、参照動画／011 NVFP4／B1／R2／R1 の5枚を並べる。

### 2. D（CLIPCached）

1. `files\q020_d1_api.json` を送る（キャッシュが無い状態）。
2. ComfyUIを再起動する（タスク `MuseLaunchComfyUI` を止めて、もう一度起動する）。
3. `files\q020_d2_api.json` を送る（ディスクのキャッシュから読む想定）。
4. 2回それぞれで、次の3点を記録する。
   - `first_progress_sec`
   - `Prompt executed in`
   - ログに `Requested to load MiniMaxH3TEModel_` の行があるかどうか（d2で無ければ、キャッシュから読めている）
5. 比べる相手は、B1計測回の `first_progress_sec`（116.1秒、ネイティブのノード）。

### 3. Qwenの img2img（2本・最後に）

018の (イ) が使えるかを見るための試験。

1. 015の `a2_meas.png`（Turbo版、seed 2026111602）を、`C:\AI model\ComfyUI\input\q020_turbo_draft.png` という名前でコピーする。
2. **011／A1で使った通常のint8のグラフ**をもとに、次の2点だけを変える。
   - 潜在画像の入力を、空の潜在画像（EmptyLatentImage）から、`LoadImage(q020_turbo_draft.png)` → `VAEEncode`（VAEは同じ `qwen_image_2.1_vae_bf16.safetensors`）に替える。
   - サンプラーの denoise を、1本目は **0.35**、2本目は **0.5** にする。KSamplerなら `denoise`、BasicSchedulerを使っているなら、その `denoise` を変える。
   - ほかの設定（プロンプト、ネガティブプロンプト、参照2枚、28ステップ、cfg 1.5、euler、simple、seed 2026111602、1088×1920）は変えない。
3. 比較画像を作る。並べる順は、左から a2_meas（Turbo）、denoise 0.35、denoise 0.5、A1（通常のint8）。髪の部分を拡大した切り抜きも、同じ並びで1枚作る。

### 報告

`021-muse.md`（`waiting_for: claude`）に書く。形は015と同じ。クラッシュがあれば、その記録も入れる。

## 3. 本人の判断が要ること

- **スパース無しでも落ちた場合に、`--disable-comfy-compiler` を本番の起動引数に入れるかどうか。** 入れれば今回の落ち方は起きなくなるが、速度が少し変わるかもしれない。入れる場合は、その前に1本だけ速度を比べる。
- **クラッシュダンプ（約780MB）を消すかどうか。** 場所は `C:\Users\user\AppData\Local\CrashDumps\python.exe.24120.dmp`。
- R1・R2・Qwen img2img の結果を見て、どれを採用するか。
