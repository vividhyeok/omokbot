# 오목봇

플레이할수록 사용자의 습관을 읽는 단일 메인 오목 AI입니다. 포지셔닝은 "강화학습계의 Teachable Machine"에 가깝고, 학습 체감이 먼저 보이도록 설계된 게임형 체험 프로젝트입니다. finite/infinite 분기는 제거했고, 루트에서 바로 실행되는 클라이언트 사이드 게임만 남겼습니다.

## 무엇이 들어 있나

- 휴리스틱 수읽기와 Minimax 기반 기본 엔진
- TensorFlow.js 적응 레이어
- 사용자 다음 수 예측 히트맵
- 봇 후보 확률 히트맵
- 실시간 승률 그래프
- 게임별 Loss 기록
- 로컬 스토리지 저장/불러오기

## 시각화 모드

하단의 시각화 토글은 4단계로 동작합니다.

1. `시각화: 끄기`
2. `AI 뇌구조 맵 보기`
3. `유저 수 예측 보기`
4. `봇 후보 확률 보기`

유저 수 예측과 봇 후보 확률은 보드 위 히트맵으로 확인하는 방식입니다. 텍스트 패널은 의도적으로 제거했습니다.

## 초기화 동작

초기화 버튼은 저장된 기억을 지우고, 대국을 중단한 뒤 시작 오버레이로 돌아갑니다. 새 게임을 시작하기 전의 첫 화면으로 복귀합니다.

## 저장 데이터

| 키 | 내용 |
|---|---|
| `gomoku-adaptive-weights` | 적응 레이어 모델 가중치 |
| `gomoku-player-model` | 사용자 행동/전이/위협 분석 |
| `gomoku-history` | 진행 중 기보 |
| `gomoku-stats` | 전적 |
| `gomoku-loss-history` | 학습 손실 추이 |
| `gomoku-winprob-history` | 판별 승률 추이 |

## 실행

브라우저에서 [index.html](index.html)을 열면 바로 실행됩니다. 별도 서버나 빌드는 필요 없습니다.

## 정리된 파일

- 메인 실행: [game.js](game.js), [index.html](index.html), [style.css](style.css)
- 삭제된 버전 폴더: `finite/`, `infinite/`
- 삭제된 과거/임시 파일: `old_game.js`, `old_game_utf8.js`, `temp_4a.js`, `temp_8c.js`, `temp_dd.js`
