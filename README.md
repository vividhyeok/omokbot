# 적응형 오목 (Adaptive Gomoku)

[🇰🇷 한국어](#한국어) | [🇺🇸 English](#english)

---

<a id="한국어"></a>
## 🇰🇷 한국어 버전

> 플레이할수록 당신을 연구하는 오목 AI

### 프로젝트 개요

이 프로젝트는 단순한 오목 게임이 아닙니다.

핵심 목표는 **"AI가 학습하고 있다는 사실을 사용자가 직접 눈으로 볼 수 있게 만드는 것"** 입니다. 구글의 티처블 머신(Teachable Machine)처럼, 처음부터 어느 정도 게임이 가능한 수준으로 동작하되 — 판을 거듭할수록 봇이 당신의 패턴을 분석하고, 약점을 파악하고, 그에 맞는 전략으로 점점 까다롭게 변해갑니다. 그 과정을 숫자와 그래프로 실시간 공개합니다.

학습은 완전히 브라우저 안에서 이루어집니다. 서버도 없고, 계정도 없고, 외부 데이터 전송도 없습니다. 모든 기억은 당신의 브라우저 로컬 스토리지에만 저장됩니다. 이 프로젝트는 **Vercel 무료 티어를 통해 배포**되어 누구나 쉽게 접근 가능합니다.

### 만들게 된 이유

강화학습이라는 개념은 많이 알려져 있지만, 실제로 학습이 일어나는 과정을 체감하기는 어렵습니다. 대부분의 AI 게임 봇은 "이미 완성된 상태"로 제공되고, 내부에서 무슨 일이 일어나는지는 보이지 않습니다.

반면 티처블 머신이 인상적인 이유는, 학습 과정 자체가 인터페이스가 되기 때문입니다. 데이터를 넣고, 학습을 누르고, 결과가 바뀌는 걸 바로 확인하는 — 그 흐름이 강화학습을 직관적으로 이해하게 만들어줍니다.

오목은 그 아이디어를 게임 형태로 구현하기에 적합한 도메인입니다. 규칙이 단순하고, 수가 쌓이면 패턴이 생기고, 패턴은 데이터가 됩니다. 게임을 하면 할수록 봇에게 데이터를 주는 셈이고, 봇은 그 데이터로 당신을 상대하는 법을 조금씩 배워갑니다.

이 프로젝트는 그 과정을 **투명하게 보여주는 것**에 집중합니다.

### 주요 기능

#### 두 층으로 이루어진 봇 구조
봇은 두 가지 레이어로 동작합니다.
- 첫 번째는 휴리스틱 기반의 베이스 레이어입니다. 열린 4, 열린 3, 양방향 막기 등 오목의 전형적인 패턴을 점수로 환산하고, Minimax 알고리즘과 Alpha-Beta 가지치기로 최선의 수를 선택합니다. 이 레이어는 변하지 않습니다. 게임 1판부터 중급 이상의 수준을 유지합니다.
- 두 번째는 TensorFlow.js 기반의 적응 레이어입니다. 당신과의 게임 결과를 바탕으로 브라우저 안에서 직접 파인튜닝됩니다. 이 레이어가 베이스 레이어의 점수에 modifier를 곱하는 방식으로 개입하며, 게임이 쌓일수록 그 비중이 커집니다. 3~5판이면 체감 가능한 변화가 시작됩니다.

#### 다음 수 예측 히트맵
봇이 "당신이 다음에 어디에 둘 것 같은가"를 실시간으로 평가합니다. 가능성이 높은 칸일수록 진한 빨강으로 표시됩니다. 단일 빨강 계열의 농담 차이만으로 표현하기 때문에, 시각적으로 산만하지 않으면서도 봇의 예측 집중도를 직관적으로 읽을 수 있습니다. 상위 3칸에는 퍼센트 비율도 표시됩니다.

#### 실시간 승률 계산
매 착수마다 현재 국면에서의 승률을 재계산해 보여줍니다. 좋은 수를 두면 파란 선이 올라가고, 실수하면 뚝 떨어집니다. 이번 판에서 가장 크게 역전된 수에는 별도 마커가 표시됩니다. 학습이 쌓일수록 같은 국면에서 봇의 승률 수치가 자연스럽게 높아집니다 — 이것이 학습 효과가 숫자로 드러나는 방식입니다.

#### 학습 지표 시각화
사이드 패널에 다음 정보가 실시간으로 표시됩니다.
- 이번 수에서 적응 레이어가 기여한 비율 (%)
- 봇이 탐색한 후보 수의 개수
- 봇의 확신도 (Low / Mid / High)
- 게임별 Training Loss 스파크라인 — 아래로 내려갈수록 학습 중
- 역대 게임별 봇 평균 승률 추이 — 위로 올라갈수록 강해지는 중

#### 플레이어 스타일 분석
봇은 당신의 패턴도 분류합니다. 어느 방향을 선호하는지, 귀와 중앙 중 어디에 집중하는지, 공격형인지 수비형인지, 열린 3을 막는 데 얼마나 실패하는지. 이 데이터는 레이더 차트로 시각화되고, 자연어 태그로도 표시됩니다. 봇은 이 분석을 바탕으로 전략을 조정합니다.

#### 수 품질 피드백
당신이 돌을 놓을 때마다, 그 수가 최선 대비 얼마나 좋은 수였는지 짧게 알려줍니다. 최선의 수를 뒀다면 초록 뱃지가, 놓쳤다면 더 좋았을 칸이 잠깐 강조됩니다. 봇만 학습하는 게 아니라 당신도 같이 성장하는 구조입니다.

### 기술적 특징

- 백엔드 없음 — 완전한 클라이언트 사이드 동작
- 빌드 도구 없음 — `index.html` 파일 하나를 브라우저에서 열면 실행됩니다
- 모든 학습과 데이터는 브라우저 로컬 스토리지에만 저장됩니다
- 로컬 스토리지를 초기화하면 봇의 기억이 완전히 리셋됩니다

#### 로컬 스토리지 구조
| 키 | 저장 내용 |
|---|---|
| `gomoku-adaptive-weights` | TF.js 적응 레이어 모델 가중치 |
| `gomoku-history` | 최근 20판 착수 기록 및 결과 |
| `gomoku-stats` | 총 전적 (게임 수 / 승 / 패) |
| `gomoku-player-profile` | 플레이어 스타일 분석 데이터 |
| `gomoku-loss-history` | 게임별 훈련 Loss 기록 |
| `gomoku-winprob-history` | 게임별 봇 평균 승률 기록 |

### 실행 방법

`index.html` 파일을 Chrome, Firefox, Safari 등 최신 브라우저에서 열면 바로 실행됩니다. 별도 설치나 서버 구동이 필요 없습니다.

봇 기억을 초기화하고 싶을 때는 하단의 "기억 초기화" 버튼을 사용하거나, 브라우저 개발자 도구에서 해당 도메인의 로컬 스토리지를 직접 삭제하면 됩니다. 혹은, "이어 하기"를 통해 다른 기기에서 JSON 파일을 가져와 연동할 수도 있습니다.

### 앞으로의 방향

현재 구현은 개인 플레이어와의 1:1 적응에 집중합니다. 이후 가능한 확장 방향으로는 플레이어 간 스타일 비교, 봇의 학습 곡선 내보내기, 복수의 플레이어 프로필 지원 등을 고려하고 있습니다.

---

<a id="english"></a>
## 🇺🇸 English Version

> A Gomoku AI that studies you as you play

### Project Overview

This project is not just a simple Gomoku game.

The core objective is to **"make the fact that the AI is learning visibly apparent to the user in real-time"**. Like Google's Teachable Machine, the bot starts with a baseline level of competence—but as you play more games, the bot analyzes your patterns, identifies your weaknesses, and gradually shifts its strategy to counter you. This process is displayed in real-time through numbers and charts.

All learning happens entirely within the browser. There is no server, no accounts, and no external data transmission. All "memories" are stored exclusively in your browser's local storage. This project is intended to be **deployed via Vercel's free tier**.

### Motivation

While the concept of Reinforcement Learning is well known, practically experiencing the learning process is rare. Most AI game bots are delivered as "finished products," leaving what happens under the hood invisible.

Teachable Machine is impressive because the learning process itself becomes the UI. Feeding data, hitting train, and immediately seeing the output change—this loop makes ML intuitively understandable.

Gomoku is a perfect domain to adapt this idea into a game. The rules are simple, sequential moves create clear patterns, and those patterns become training data. The more you play, the more training data you feed the bot, and the better it gets at defeating you.

This project focuses on **making that process transparent**.

### Key Features

#### Two-Layer Bot Architecture
The bot operates on two layers:
- The first is a heuristic-based baseline layer. It assigns points to typical Gomoku patterns (open 4, open 3, dual-blocks) and uses Minimax with Alpha-Beta pruning to find the best move. This layer never changes, ensuring mid-to-high difficulty starting from Game 1.
- The second is the adaptive layer powered by **TensorFlow.js**. It is fine-tuned directly in your browser based on your win/loss results. This layer intervenes by multiplying a modifier to the baseline scores. As games accumulate, its influence grows. You will begin to notice changes in strategy by games 3–5.

#### Predictive Move Heatmap
The bot evaluates "where you are most likely to place your next stone" in real-time. Highly probable cells are shaded deep red. By relying purely on redness intensity, you can intuitively read the bot's prediction confidence without UI clutter. Top 3 predictions display their exact percentages.

#### Real-Time Win Probability
After every move, the win probability of the current board state is recalculated. Great moves spike the blue line; bad ones crash it. The most critical turning point in the match gets explicitly marked. As the bot learns over multiple games, you'll see the bot's native probability estimate for the same board state naturally trend upward—this is learning manifested as a metric.

#### Visualized Learning Metrics
The side panel displays the following real-time data:
- How much the Adaptive Layer influenced the current move (%)
- Number of candidate moves the bot evaluated
- Bot's confidence level (Low / Mid / High)
- Training Loss sparkline across games (downward trend = learning)
- Average bot win probability across games (upward trend = getting stronger)

#### Player Style Profiling
The bot classifies your habits: whether you prefer corners over the center, favor aggressive lines or defensive blocks, or fail to catch open 3s. This is plotted onto a dynamic SVG radar chart along with natural-language tendency tags. The bot uses this exact profile to pivot its neural predictions.

#### Move Quality Feedback
Whenever you drop a stone, a micro-badge lets you know how your move measured up against the mathematically best possible move. Catching a perfect block shows a green badge; missing a lethal threat highlights the cell you *should* have played instead.

### Technical Details

- No Backend — 100% Client-side operation
- No Build Tools — Runs natively immediately upon opening `index.html`
- All ML fine-tuning and state saves exclusively inside `localStorage`
- Neural memories synchronize cross-device via pure JSON file exporting/importing.

#### Local Storage Schema
| Key | Contents |
|---|---|
| `gomoku-adaptive-weights` | TF.js model weights representing the adaptive neural layer |
| `gomoku-history` | Replay log of the last 20 matches |
| `gomoku-stats` | Total wins, draws, and losses |
| `gomoku-player-profile` | Derived player style metrics (aggression, center bias, etc.) |
| `gomoku-loss-history` | Training loss metric values per game |
| `gomoku-winprob-history` | Average bot win probability metrics per game |

### How to Run

Simply open `index.html` in any modern browser (Chrome, Firefox, Safari). No installation or server required.

To initialize a fresh bot, click the "초기화" (Reset) button or manually clear the domain's local storage in Developer Tools. You can also transfer a bot's "brain" between your PC and phone using the "이어 하기" (Import JSON) feature.

### Future Roadmap

The current implementation focuses strictly on 1:1 local adaptation. Future features may include exporting the raw learning curve to a CSV, allowing player-vs-player style analysis, and maintaining multiple distinct player profiles in the same browser session.
