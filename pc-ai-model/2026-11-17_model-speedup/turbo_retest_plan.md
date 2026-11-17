---
from: claude
date: 2026-11-17T15:55+09:00
waiting_for: muse
---

本人の決定（2026-11-17 15:38）:

1. H3動画のTurbo LoRAを、条件を正しく揃えて再試験する。
2. QwenのTurboは「下書き・候補出しはTurbo、本番の仕上げは通常のint8」という使い方の方向で考える。本番への組み込みはまだしない。

## 1. 質問への答え

### 017の訂正（Claudeの見立てが誤っていた）

017で「014の指示が `MiniMaxH3SigmaShift` を落としていた」と書いたのは**誤り**。PCのComfyUIの実行履歴（`/history`、`autopilot\outbox\files\setup3_02_history\comfy_history.json`）で、B1が実際に送ったグラフを確認した。

- B1には `MiniMaxH3SigmaShift`（video 12.0 / audio 3.0）が入っていた。KSampler も、そのノードを通したMODELを使っていた。
- 配布元（lightx2v）のHugging Faceの告知では、**Ref2Vの8ステップ768p LoRAの推奨は video shift 12 / audio shift 3**。shift 6 の推奨は FL2V 用で、「REF2Vには当てはまらない」と明記されている。
- つまりB1は、推奨どおりの条件（8ステップ、shift 12/3、euler、simple、cfg 1、強度1.0）で回っていた。017で挙げた「shift 6/3 と 12/3 の2通り」の試験は、根拠が無いので**取り消す**。
- LoRAの読み込みにも問題は無かった。`comfyui_run.log` に `lora key not loaded` の警告は無く、`208 patches attached` と出ている。

ここから残る原因の候補は、次の2つ（どちらも推測）:

- (a) 本体がNVFP4（4bit）であること。LoRAはbf16の本体で学習されている。
- (b) LoRAの版。v1.1は配布元が推奨値を明記していない。v1.0には推奨値の告知がある。

参照動画の動きをなぞる使い方そのものがTurboに向かない可能性（推測）もある。これは(a)(b)の結果を見てから判断する。

### B2（スパースアテンション）でログから分かったこと

- 1ステップが約123〜128秒で、B1（約97秒）より**遅くなった**。
- 終了時に、`comfy_aimdo` のメモリ解放処理で「Windows fatal exception: stack overflow」が出ていた（`Exception ignored` 扱いで、生成自体は完走している）。
- 現時点で、スパースアテンションは本番に入れない方向。

## 2. Museがやること

### 再試験（R1・R2）

- **前提:** `autopilot\outbox\setup3_03_dl_ref2v_turbo_v10.result.json` が `"status": "success"` になるまで待つ。v1.0のLoRAを、Claudeがランナー経由でダウンロード中。
- **グラフ:** このスレッドの `files\q018_r1_api.json` と `files\q018_r2_api.json` を、そのまま `/prompt` に送る。
  - どちらも、ComfyUIの履歴から取り出した **B1（q014_b1_meas）のグラフそのもの**で、違いは1か所だけ（下の表）。
  - **中身は書き換えないこと。** BlockSparseAttention は入っていない。

| 回 | 本体（node 1） | LoRA（node 14） | B1からの違い |
|---|---|---|---|
| R2（先に実行） | `minimax_h3_ref2va_pruned_nvfp4.safetensors` | `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` | LoRAの版だけ |
| R1 | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors` | 本体の精度だけ |

- 順番は R2 → R1。今NVFP4が読み込まれている状態のまま、先にR2を回すため。
- 捨て回しはしない。秒数には読み込み時間が含まれるので、その旨を書き添える。
- 実行前に、両方のファイルが `models\loras` と `models\diffusion_models` にあることを確かめる。
- 各回の実行後、ComfyUIログに `lora key not loaded` があるかを確かめて書く。

### 記録・生成物

- 各回で記録するのは、s/it、`Prompt executed in`、最大Dedicated、最大Shared（006の方法）。
- **2秒地点と5秒地点で、それぞれ5枚を横に並べた画像**を作る。並べる順は次のとおり。
  1. 参照動画 `opa_ref.mp4`
  2. 011 NVFP4（LoRAなし、`h3_ref2va_nvfp4.mp4`）
  3. B1
  4. R2
  5. R1

  参照動画の動きをなぞれているかを、本人が見比べるため。
- R1・R2の動画を `files/` に置く。

### 017の指示との関係

- 017の「B2をLoRAなしで回す」は、**R1・R2が終わってから、時間があれば1回だけ**にする。20ステップで40分以上かかる見込み。
- 途中で「stack overflow」がまた出たら、そのログの行を貼って中止してよい。
- C1・Dの結果がすでに出ていれば、それも次の報告にまとめる。

### 報告

- `019-muse.md`（`waiting_for: claude`）に書く。017で `018-muse.md` に書くよう指示していたが、この発言が018になったので、番号を1つずらす。
- **すでに `018-muse.md` を書き終えていた場合は、それはそのまま残し、R1・R2の結果を `019-muse.md` に書く。**

## 3. 本人の判断が要ること

- **R1・R2の結果を見て、Turbo LoRAを採用するか、見送りで確定するか。**
- **QwenのTurboの使い方。** Turboと通常版は別のモデルなので、同じseedでも同じ絵にはならない。「Turboで選んだ1枚を、通常版で仕上げる」には、次のどちらかになる。
  - (ア) 気に入ったプロンプトだけを、通常版で作り直す。構図は変わる。
  - (イ) Turboの絵を下敷きにして、通常版で弱めに描き直す（img2img）。構図を保てるが、まだ試していない。
- 補足（推測）: Turboは cfg 1 で動かしているので、本番のネガティブプロンプト（「grainy skin」「rough sandy texture」「noise」など）が効いていない。本人が指摘した「ざらつき」の一因かもしれない。
