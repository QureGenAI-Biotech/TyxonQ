# TyxonQ パルスインターフェース使用ガイド

## 目次

- [概要](#概要)
- [TQASM 0.2 構文仕様](#tqasm-02-構文仕様)
- [コアコンポーネント](#コアコンポーネント)
- [波形パラメータ概要表](#波形パラメータ概要表)
- [波形パラメータ詳細説明](#波形パラメータ詳細説明)
- [使用方法](#使用方法)
- [TQASM 出力形式](#tqasm-出力形式)
- [高度な機能](#高度な機能)
- [ベストプラクティス](#ベストプラクティス)
- [トラブルシューティング](#トラブルシューティング)
- [実用的なアプリケーション例](#実用的なアプリケーション例)
- [まとめ](#まとめ)

---

## 概要

TyxonQは、量子ビットのパルス信号を直接操作して精密な量子制御を実現する強力なパルスレベル制御インターフェースを提供します。パルスインターフェースを通じて、以下のことが可能です：

- カスタムパルス波形の定義
- 量子ビットキャリブレーションプログラムの作成
- 高度な量子制御アルゴリズムの実装
- TQASM 0.2形式のパルスレベル回路の生成

### サポートされている波形タイプ

現在、以下の4つの主要な波形タイプをサポートしています：
- **cosine_drag** - 漏洩状態遷移を抑制する余弦DRAG波形
- **flattop** - 量子状態準備に適した平頂波形
- **gaussian** - 滑らかなパルス遷移を提供するガウス波形
- **sine** - 周期的振動実験に使用する正弦波形

詳細なパラメータ定義と数学的表現については、下記の波形パラメータ詳細説明セクションを参照してください。

## TQASM 0.2 構文仕様

### 構文定義

TQASM 0.2は、バッカス・ナウア記法（BNF）を使用して構文構造を定義します：

```
<pulse> ::= <defcal>

<defcal> ::= "defcal" <id> <idlist> { <calgrammar> }

<calgrammar> ::= <calstatement>
               | <calgrammar> <calstatement>

<calstatement> ::= <framedecl>
                | <waveformplay>

<framedecl> ::= "frame" <id> "=" "newframe" (<idlist>);

<waveformplay> ::= "play" (<id>, <waveform>);

<waveform> ::= <id> (<explist>)
```

### キーワード説明

| キーワード | 機能説明 | 構文形式 |
|------------|----------|----------|
| `defcal` | カスタムパラメータ化波形量子ゲートの定義 | `defcal <ゲート名> <パラメータリスト> { <キャリブレーション文> }` |
| `frame` | 変数をフレームタイプとして宣言 | `frame <フレーム名> = newframe(<量子ビット>);` |
| `newframe` | ターゲット量子ビット上に波形を運ぶための新しいフレームを作成 | `newframe(<量子ビット識別子>)` |
| `play` | 指定されたフレーム上で波形を再生 | `play(<フレーム名>, <波形関数>(<パラメータ>));` |

### サポートされている波形タイプ

現在サポートされている波形関数には以下が含まれます：
- `cosine_drag(duration, amp, phase, alpha)` - 余弦DRAG波形
- `flattop(duration, amp, width)` - 平頂波形
- `gaussian(duration, amp, sigma, angle)` - ガウス波形
- `sin(duration, amp, phase, freq, angle)` - 正弦波形

### 完全な例

以下は、パラメータ化波形の定義と使用方法を示す完全なTQASM 0.2コード例です：

```tqasm
TQASM 0.2;
QREG q[1];

defcal hello_world a {
  frame drive_frame = newframe(a);
  play(drive_frame, cosine_drag(50, 0.2, 0.0, 0.0));
}

hello_world q[0];
```

### コード解析

1. **TQASM 0.2;** - TQASM 0.2バージョンの使用を宣言
2. **QREG q[1];** - 1量子ビットの量子レジスタを定義
3. **defcal hello_world a { ... }** - パラメータ"a"を持つ"hello_world"という名前のキャリブレーションプログラムを定義
4. **frame drive_frame = newframe(a);** - 量子ビット"a"上に"drive_frame"という名前のフレームを作成
5. **play(drive_frame, cosine_drag(50, 0.2, 0.0, 0.0));** - フレーム上で余弦DRAG波形を再生
6. **hello_world q[0];** - 量子ビットq[0]上でキャリブレーションプログラムを呼び出し

### 波形パラメータ説明

`cosine_drag(50, 0.2, 0.0, 0.0)` のパラメータの意味：
- `50` - パルス持続時間（サンプリング周期）
- `0.2` - 波形振幅
- `0.0` - 位相角（ラジアン）
- `0.0` - DRAG係数

## コアコンポーネント

TyxonQパルスインターフェースのコアコンポーネントには、波形タイプ、パラメータ化サポート、キャリブレーションビルダーが含まれます。これらのコンポーネントは協調して動作し、ユーザーに完全なパルス制御機能を提供します。

---

### 1. 波形タイプ

TyxonQは、特定のパラメータを持つ複数の事前定義されたパルス波形タイプをサポートします：

#### ガウス波形
```python
from tyxonq import waveforms

# ガウス波形の作成：振幅、持続時間、標準偏差
gaussian_wf = waveforms.Gaussian(amp=0.5, duration=100, sigma=20)
```

#### ガウス方形波
```python
# ガウス方形波の作成：振幅、持続時間、標準偏差、幅
gaussian_square_wf = waveforms.GaussianSquare(amp=0.5, duration=100, sigma=20, width=60)
```

#### DRAG波形
```python
# DRAG波形の作成：振幅、持続時間、標準偏差、βパラメータ
drag_wf = waveforms.Drag(amp=0.5, duration=100, sigma=20, beta=0.5)
```

#### 定数波形
```python
# 定数波形の作成：振幅、持続時間
constant_wf = waveforms.Constant(amp=0.5, duration=100)
```

#### 正弦波形
```python
# 正弦波形の作成：振幅、周波数、持続時間
sine_wf = waveforms.Sine(amp=0.5, frequency=0.1, duration=100)
```

#### 余弦波形
```python
# 余弦波形の作成：振幅、周波数、持続時間
cosine_wf = waveforms.Cosine(amp=0.5, frequency=0.1, duration=100)
```

#### 余弦DRAG波形
```python
# 余弦DRAG波形の作成：振幅、持続時間、位相、αパラメータ
cosine_drag_wf = waveforms.CosineDrag(amp=0.5, duration=100, phase=0.0, alpha=0.2)
```

#### 平頂波形
```python
# 平頂波形の作成：振幅、幅、持続時間
flattop_wf = waveforms.Flattop(amp=0.5, width=60, duration=100)
```

### 2. パラメータ化サポート

すべての波形は、`Param`クラスを使用したパラメータ化をサポートします：

```python
from tyxonq import Param

# パラメータ化波形の作成
param_t = Param("t")
parametric_wf = waveforms.CosineDrag(param_t, 0.2, 0.0, 0.0)
```

### 3. キャリブレーションビルダー

`DefcalBuilder`は、量子ビットキャリブレーションプログラムを作成するためのコアツールです：

```python
from tyxonq import Circuit, Param

# 回路の作成とパルスモードの有効化
qc = Circuit(1)
qc.use_pulse()

# パラメータの作成
param0 = Param("a")

# キャリブレーションプログラムの構築開始
builder = qc.calibrate("calibration_name", [param0])

# フレームの定義
builder.new_frame("drive_frame", param0)

# 波形の再生
builder.play("drive_frame", waveforms.CosineDrag(param0, 0.2, 0.0, 0.0))

# キャリブレーションプログラムの構築
builder.build()
```

## 波形パラメータ概要表

以下の表は、サポートされているすべての波形のクイックリファレンスを提供し、パラメータ形式と主要なアプリケーションシナリオを含みます：

| 番号 | 波形タイプ | 波形パラメータ | 主要用途 |
|------|------------|----------------|----------|
| 1 | `cosine_drag` | `CosineDrag(duration, amp, phase, alpha)` | 漏洩状態遷移抑制のための精密制御 |
| 2 | `flattop` | `Flattop(duration, amp, width)` | 量子状態準備のための平頂パルス |
| 3 | `gaussian` | `Gaussian(duration, amp, sigma, angle)` | 滑らかな遷移のためのガウスパルス |
| 4 | `sin` | `Sin(duration, amp, phase, freq, angle)` | 周期的振動のための正弦パルス |
| 5 | `drag` | `Drag(duration, amp, sigma, beta)` | 超伝導量子ビット制御のためのDRAGプロトコル |
| 6 | `constant` | `Constant(duration, amp)` | DCバイアスのための定数パルス |
| 7 | `gaussian_square` | `GaussianSquare(duration, amp, sigma, width)` | ガウスエッジ方形波 |
| 8 | `cosine` | `Cosine(duration, amp, freq, phase)` | 余弦パルス |

## 波形パラメータ詳細説明

このセクションでは、各波形のパラメータ定義、数学的表現、物理的意味を詳細に説明します。各波形には特定のアプリケーションシナリオとパラメータ制約があります。

---

### 1. CosineDrag波形パラメータ

| パラメータ | タイプ | 説明 | 制約条件 |
|------------|--------|------|----------|
| `amp` | 実数値 | 波形振幅 | \|amp\| ≤ 2 |
| `duration` | 整数 | パルス長（サンプリング周期単位） | 0 < duration < 10000 |
| `phase` | 実数値 | 位相角（ラジアン） | 特別な制限なし |
| `alpha` | 実数値 | DRAG係数 | \|alpha\| ≤ 10 |

**数学的表現**: 
- `g(x) = (Amp / 2) × e^(i × phase) × [cos((2πx / duration) - π) + 1]`
- `output(x) = g(x) + i × alpha × g'(x)`
- 定義域: `x ∈ [0, duration)`

**パラメータ説明**: 
- `amp`: 波形振幅、波形の強度を制御
- `duration`: サンプリング周期単位でのパルス持続時間
- `phase`: 位相角、波形の位相オフセットを制御
- `alpha`: 漏洩状態遷移を抑制するためのDRAG係数

### 2. Flattop波形パラメータ

| パラメータ | タイプ | 説明 | 制約条件 |
|------------|--------|------|----------|
| `amp` | 実数値 | 波形振幅 | amp ≤ 2 |
| `width` | 実数値 | ガウス成分の半値全幅（FWHM） | width ≤ 100 |
| `duration` | 整数 | パルス長（サンプリング周期単位） | duration ≤ 100,000 |

**数学的表現**: 
- `w = width` （ガウス成分の半値全幅）
- `σ = w / √(4 log 2)` （標準偏差）
- `A = amp` （振幅）
- `T = duration` （持続時間）
- `output(x) = (A / 2) × [erf((w + T - x) / σ) - erf((w - x) / σ)]`
- 定義域: `x ∈ [0, T + 2w)`

**パラメータ説明**: 
- `amp`: 波形振幅、波形の全体的な強度を制御
- `width`: ガウス成分の半値全幅、ガウスエッジの幅を制御
- `duration`: パルス持続時間、平頂部分の長さを制御

### 3. ガウス波形パラメータ

| パラメータ | タイプ | 説明 | 制約条件 |
|------------|--------|------|----------|
| `amp` | 実数値 | 波形振幅 | \|amp\| ≤ 2 |
| `duration` | 整数 | パルス長（サンプリング周期単位） | 0 < duration < 10000 |
| `sigma` | 実数値 | ガウス波形の標準偏差 | 特別な制限なし |
| `angle` | 実数値 | 複素位相因子の角度（ラジアン） | 特別な制限なし |

**数学的表現**: 
- `f'(x) = exp(- (1/2) × ((x - duration/2)² / sigma²))`
- `f(x) = A × f'(x)` ただし `0 ≤ x < duration`
- `A = amp × exp(i × angle)`

**パラメータ説明**: 
- `amp`: 波形振幅、波形の強度を制御
- `duration`: サンプリング周期単位でのパルス持続時間
- `sigma`: ガウス分布の標準偏差、波形の幅を制御
- `angle`: 複素位相因子、波形の位相を制御

### 4. 正弦波形パラメータ

| パラメータ | タイプ | 説明 | 制約条件 |
|------------|--------|------|----------|
| `amp` | 実数値 | 正弦波振幅、波形範囲 [-amp, amp] | \|amp\| ≤ 2 |
| `phase` | 実数値 | 正弦波の位相（ラジアン） | 特別な制限なし |
| `freq` | 実数値 | 正弦波周波数（サンプリング周期の逆数） | 特別な制限なし |
| `angle` | 実数値 | 複素位相因子の角度（ラジアン） | 特別な制限なし |
| `duration` | 整数 | パルス長（サンプリング周期単位） | 0 < duration < 10000 |

**数学的表現**: 
- `f(x) = A sin(2π × freq × x + phase)` ただし `0 ≤ x < duration`
- `A = amp × exp(i × angle)`

**パラメータ説明**: 
- `amp`: 正弦波振幅、波形の強度を制御、範囲 [-amp, amp]
- `phase`: 正弦波の位相、波形の位相オフセットを制御
- `freq`: サンプリング周期の逆数での正弦波周波数
- `angle`: 複素位相因子、複素波形位相を制御
- `duration`: サンプリング周期単位でのパルス持続時間

### 5. その他の波形パラメータ

#### GaussianSquare波形
| パラメータ | タイプ | 説明 | 制約条件 |
|------------|--------|------|----------|
| `amp` | 実数値 | 波形振幅 | \|amp\| ≤ 2 |
| `duration` | 整数 | パルス長（サンプリング周期単位） | 0 < duration < 10000 |
| `sigma` | 実数値 | ガウス成分の標準偏差 | 特別な制限なし |
| `width` | 実数値 | 方形波部分の幅 | width ≤ duration |

#### Drag波形
| パラメータ | タイプ | 説明 | 制約条件 |
|------------|--------|------|----------|
| `amp` | 実数値 | 波形振幅 | \|amp\| ≤ 2 |
| `duration` | 整数 | パルス長（サンプリング周期単位） | 0 < duration < 10000 |
| `sigma` | 実数値 | ガウス成分の標準偏差 | 特別な制限なし |
| `beta` | 実数値 | 漏洩状態遷移を抑制するためのDRAGパラメータ | 特別な制限なし |

**数学的表現**: 
- `f(x) = A × exp(-(x - duration/2)² / (2 × sigma²))` ただし `0 ≤ x < duration`
- `A = amp × exp(i × angle)` （angleパラメータがサポートされている場合）

#### 定数波形
| パラメータ | タイプ | 説明 | 制約条件 |
|------------|--------|------|----------|
| `amp` | 実数値 | 定数振幅 | \|amp\| ≤ 2 |
| `duration` | 整数 | パルス長（サンプリング周期単位） | 0 < duration < 10000 |

**数学的表現**: 
- `f(x) = amp` ただし `0 ≤ x < duration`
- 定義域: `x ∈ [0, duration)`

**パラメータ説明**: 
- `amp`: 定数振幅、持続時間全体で一定
- `duration`: サンプリング周期単位でのパルス持続時間

## 使用方法

TyxonQパルスインターフェースは、直感的で使いやすいAPIを提供し、ユーザーが複雑なパルス制御プログラムを簡単に作成できるようにします。

---

### 基本的なワークフロー

1. **パルスモードの有効化**
```python
qc = Circuit(n_qubits)
qc.use_pulse()
```

2. **キャリブレーションプログラムの定義**
```python
# DefcalBuilderを使用してキャリブレーションプログラムを構築
builder = qc.calibrate("cal_name", [param1, param2])
builder.new_frame("frame_name", qubit_param)
builder.play("frame_name", waveform)
builder.build()
```

3. **キャリブレーションプログラムの呼び出し**
```python
# 回路内でキャリブレーションプログラムを呼び出し
qc.add_calibration('cal_name', ['q[0]'])
```

4. **TQASMコードの生成**
```python
tqasm_code = qc.to_tqasm()
```

### 完全な例

#### 例1：簡単なラビ振動実験

```python
import sys
import os
sys.path.insert(0, "..")

from tyxonq import Circuit, Param, waveforms
from tyxonq.cloud import apis

def create_rabi_circuit(t):
    """ラビ振動実験回路の作成"""
    qc = Circuit(1)
    qc.use_pulse()
    
    # パラメータの作成
    param_t = Param("t")
    
    # キャリブレーションプログラムの構築
    builder = qc.calibrate("rabi_experiment", [param_t])
    builder.new_frame("drive_frame", param_t)
    builder.play("drive_frame", waveforms.CosineDrag(param_t, 0.2, 0.0, 0.0))
    builder.build()
    
    # キャリブレーションプログラムの呼び出し
    qc.add_calibration('rabi_experiment', ['q[0]'])
    
    return qc

# 異なる時間パラメータを持つ回路の作成
for t in [10, 30, 50, 70, 90]:
    qc = create_rabi_circuit(t)
    print(f"t={t}のTQASM:")
    print(qc.to_tqasm())
    print("-" * 50)
```

## TQASM出力形式

生成されたTQASMコードはTQASM 0.2標準に準拠します：

```tqasm
TQASM 0.2;
QREG q[1];

defcal rabi_experiment a {
  frame drive_frame = newframe(a);
  play(drive_frame, cosine_drag(a, 0.2, 0.0, 0.0));
}

rabi_experiment q[0];
```

## 高度な機能

### 1. 時間制御

波形に開始時間パラメータを追加できます：

```python
builder.play("frame_name", waveform, start_time=50)
```

### 2. 複雑なキャリブレーションプログラム

複数の命令を含む複雑なキャリブレーションプログラムを構築できます：

```python
builder = qc.calibrate("complex_cal", [param])
builder.new_frame("frame1", param)
builder.play("frame1", waveform1)
builder.new_frame("frame2", param)
builder.play("frame2", waveform2)
builder.build()
```

### 3. クラウドAPI統合

```python
from tyxonq.cloud import apis

# 認証の設定
apis.set_token("your_token")
apis.set_provider("tyxonq")

# パルス回路タスクの送信
task = apis.submit_task(
    circuit=qc,
    shots=1000,
    device="homebrew_s2",
    enable_qos_gate_decomposition=False,
    enable_qos_qubit_mapping=False,
)

# 結果の取得
result = task.results()
```

## ベストプラクティス

1. **パラメータ命名**: 理解とデバッグを容易にするため、意味のあるパラメータ名を使用
2. **波形選択**: 物理的ニーズに基づいて適切な波形タイプを選択
3. **時間単位**: 時間パラメータの単位（通常はナノ秒）に注意
4. **エラー処理**: ハードウェアに送信する前にTQASMコードの正確性を検証
5. **ドキュメント化**: 複雑なキャリブレーションプログラムにコメントと説明を追加
6. **複素位相**: 一部の波形は複素位相因子（angleパラメータ）をサポートし、精密な位相制御を実現
7. **定義域**: 各波形の定義域範囲に注意し、合理的なパラメータ設定を確保

## 波形選択ガイド

### アプリケーションシナリオに基づく波形選択

- **CosineDrag**: 漏洩状態遷移抑制を必要とする精密制御に適している（単一量子ビットゲート操作など）
- **Flattop**: 平頂パルスを必要とするアプリケーションに適している（量子状態準備など）
- **Gaussian**: 滑らかな遷移を必要とするパルスに適している（断熱進化など）
- **Sine**: 周期的振動を必要とするアプリケーションに適している（ラビ振動実験など）
- **Drag**: 超伝導量子ビットの精密制御に適している
- **Constant**: 単純な定数パルスに適している（DCバイアスなど）
- **GaussianSquare**: ガウスエッジを持つ方形波パルスに適している

## トラブルシューティング

### 一般的な問題

1. **サポートされていない波形タイプ**: 事前定義された波形タイプの使用を確認
2. **パラメータタイプエラー**: パラメータが`Param`タイプまたは数値であることを確認
3. **フレーム未定義**: 波形を再生する前にフレームが定義されていることを確認
4. **TQASM生成失敗**: キャリブレーションプログラムの構築順序を確認

## 実用的なアプリケーション例

### 例：精密なラビ振動実験

数学的定義に基づく精密なパラメータ設定：

```python
from tyxonq import Circuit, Param, waveforms

def create_precise_rabi_circuit(t_duration, amplitude, frequency):
    """
    精密なラビ振動実験回路の作成
    
    パラメータ:
    - t_duration: パルス持続時間（サンプリング周期）
    - amplitude: 正弦波振幅 (|amp| ≤ 2)
    - frequency: 正弦波周波数（サンプリング周期の逆数）
    """
    qc = Circuit(1)
    qc.use_pulse()
    
    # パラメータ化波形の作成
    param_t = Param("t")
    
    sine_wave = waveforms.Sine(
        duration=t_duration,      # 持続時間
        amp=amplitude,            # 振幅
        frequency=frequency,      # 周波数
    )

    
    # キャリブレーションプログラムの構築
    builder = qc.calibrate("precise_rabi", [param_t])
    builder.new_frame("drive_frame", param_t)
    builder.play("drive_frame", sine_wave)
    builder.build()
    
    # キャリブレーションプログラムの呼び出し
    qc.add_calibration('precise_rabi', ['q[0]'])
    
    return qc

# パラメータスキャンのための異なるパラメータを持つ回路の作成
frequencies = [0.01, 0.02, 0.05, 0.1]  # 異なる周波数
amplitudes = [0.5, 1.0, 1.5]            # 異なる振幅

for freq in frequencies:
    for amp in amplitudes:
        qc = create_precise_rabi_circuit(
            t_duration=100,    # 100サンプリング周期
            amplitude=amp,      # 振幅
            frequency=freq      # 周波数
        )
        print(f"周波数: {freq}, 振幅: {amp}")
        print(qc.to_tqasm())
        print("-" * 50)
```

### 例：DRAGパルス最適化

```python
def create_optimized_drag_pulse():
    """最適化されたDRAGパルスの作成"""
    qc = Circuit(1)
    qc.use_pulse()
    
    param_qubit = Param("q")
    
    # DRAGプロトコルを使用して漏洩状態遷移を抑制
    # f(x) = A × exp(-(x - duration/2)² / (2 × sigma²))
    # A = amp × exp(i × angle)
    drag_wave = waveforms.Drag(
        duration=100,    
        amp=1.0,         # 振幅
        sigma=20,        # ガウス標準偏差
        beta=0.5         # 漏洩状態を抑制するためのDRAGパラメータ
    )
    
    builder = qc.calibrate("optimized_drag", [param_qubit])
    builder.new_frame("drive_frame", param_qubit)
    builder.play("drive_frame", drag_wave)
    builder.build()
    
    qc.add_calibration('optimized_drag', ['q[0]'])
    return qc
```

## まとめ
