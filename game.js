// ==========================================
// 설정 및 상태
// ==========================================
const SIZE = 15, CELL = 35, PAD = 20, CANVAS_SIZE = SIZE * CELL + PAD * 2;
let board = JSON.parse(localStorage.getItem('gomoku-ongoing-board')) || Array(SIZE * SIZE).fill(0);
let gameHistory = JSON.parse(localStorage.getItem('gomoku-ongoing-history')) || []; 
let winProbTrace = JSON.parse(localStorage.getItem('gomoku-ongoing-prob')) || [0.5];

// LocalStorage 값 파싱
let stats = JSON.parse(localStorage.getItem('gomoku-stats')) || { won: 0, lost: 0, draws: 0, total: 0 }; 
let lossHistory = JSON.parse(localStorage.getItem('gomoku-loss-history')) || [];
let winProbHistory = JSON.parse(localStorage.getItem('gomoku-winprob-history')) || [];
let viewMode = 0, isThinking = false, model = null, modelReady = false;
let modelLoadToken = 0;
const VIEW_MODES = ['시각화: 끄기', 'AI 뇌구조 맵 보기', '유저 수 예측 보기', '봇 후보 확률 보기'];
let hoverPos = null; 
let lastUserMoveIdx = null;

const PLAYER_MODEL_KEY = 'gomoku-player-model';
const createDefaultPlayerModel = () => ({
    moveCount: Array(SIZE * SIZE).fill(0),
    transition: {},
    threatCells: {},
    gamesAnalyzed: 0
});

let playerModel = (() => {
    const raw = JSON.parse(localStorage.getItem(PLAYER_MODEL_KEY) || 'null');
    if (!raw || !Array.isArray(raw.moveCount) || raw.moveCount.length !== SIZE * SIZE) {
        return createDefaultPlayerModel();
    }
    return {
        moveCount: raw.moveCount,
        transition: raw.transition || {},
        threatCells: raw.threatCells || {},
        gamesAnalyzed: Number(raw.gamesAnalyzed) || 0
    };
})();

for (let i = gameHistory.length - 1; i >= 0; i--) {
    if (gameHistory[i].player === 1) {
        lastUserMoveIdx = gameHistory[i].idx;
        break;
    }
}

function savePlayerModel() {
    localStorage.setItem(PLAYER_MODEL_KEY, JSON.stringify(playerModel));
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

function setStartupStatus(text, isReady = false) {
    const status = document.getElementById('startup-status');
    const statusText = document.getElementById('startup-status-text');
    if (!status || !statusText) return;
    statusText.textContent = text;
    status.classList.toggle('is-ready', isReady);
}

function pushLog(msg, type='bot-sys') {
    const c = document.getElementById('ai-console');
    const d = document.createElement('div');
    d.className = `chat-bubble ${type}`;
    d.innerHTML = msg;
    c.appendChild(d);
    c.scrollTop = c.scrollHeight;
}

function createAdaptiveModel() {
    const adaptiveModel = tf.sequential({ layers: [
        tf.layers.reshape({ targetShape: [15, 15, 1], inputShape: [225] }),
        tf.layers.conv2d({ filters: 32, kernelSize: 5, padding: 'same', activation: 'relu' }),
        tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu' }),
        tf.layers.flatten(),
        tf.layers.dense({ units: 64, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 1, activation: 'tanh' })
    ]});
    adaptiveModel.compile({ optimizer: tf.train.adam(stats.total < 5 ? 0.005 : 0.001), loss: 'meanSquaredError' });
    return adaptiveModel;
}

const canvas = document.getElementById('board-canvas'), ctx = canvas.getContext('2d');
canvas.width = canvas.height = CANVAS_SIZE;

const PATTERNS = { WIN: 10000000, OPEN_4: 1000000, BLOCKED_4: 100000, OPEN_3: 10000, BLOCKED_3: 5000, OPEN_2: 100 };

const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

function getDifficultyProfile(totalGames) {
    if (totalGames < 5) {
        return {
            depth: 1,
            adaptiveWeight: 2500 + totalGames * 900,
            predictionWeight: 5000 + totalGames * 1300,
            noise: 0.52 - totalGames * 0.07,
            topN: 6 + totalGames
        };
    }
    if (totalGames < 8) {
        const t = (totalGames - 5) / 3;
        return {
            depth: 1 + Math.round(t),
            adaptiveWeight: Math.round(7000 + t * 26000),
            predictionWeight: Math.round(14000 + t * 42000),
            noise: Math.max(0.02, 0.22 - t * 0.18),
            topN: 10 + Math.round(t * 6)
        };
    }
    return { depth: 2, adaptiveWeight: 32000, predictionWeight: 64000, noise: 0.01, topN: 20 };
}

function sampleBySoftmax(rankedMoves, temperature) {
    if (!rankedMoves.length) return -1;
    const t = Math.max(0.01, temperature);
    const maxScore = rankedMoves[0].score;
    let probs = rankedMoves.map(m => Math.exp((m.score - maxScore) / (Math.abs(maxScore) * 0.05 + t * 10000)));
    const sum = probs.reduce((a, b) => a + b, 0);
    let r = Math.random() * sum;
    for (let i = 0; i < rankedMoves.length; i++) {
        r -= probs[i];
        if (r <= 0) return rankedMoves[i].idx;
    }
    return rankedMoves[0].idx;
}

function getSoftmaxRanked(rankedMoves, temperature) {
    if (!rankedMoves.length) return [];
    const t = Math.max(0.01, temperature);
    const maxScore = rankedMoves[0].score;
    const exps = rankedMoves.map(m => Math.exp((m.score - maxScore) / (Math.abs(maxScore) * 0.05 + t * 10000)));
    const sum = exps.reduce((a, b) => a + b, 0) || 1;
    return rankedMoves.map((m, i) => ({ ...m, prob: exps[i] / sum }));
}

async function getBotCandidateHeatmap(b) {
    let cans = getCandidates(b);
    if (!cans.length) return [];
    const profile = getDifficultyProfile(stats.total);
    let scored = [];

    for (let idx of cans) {
        b[idx] = 2;
        const rawScore = evaluateBoard(b, 2);
        b[idx] = 0;
        scored.push({ idx, rawScore });
    }

    scored.sort((a, b) => b.rawScore - a.rawScore);
    scored = scored.slice(0, profile.topN);

    const topIndices = scored.map(c => c.idx);
    const mods = await getAdaptiveModifiersForCandidates(b, topIndices);
    const predictionMap = getUserPredictionMap(b);
    const ranked = scored.map((item, i) => {
        const predictionScore = predictionMap[item.idx] || 0;
        const blended = item.rawScore + (mods[i] * profile.adaptiveWeight) + (predictionScore * profile.predictionWeight);
        return { idx: item.idx, score: blended };
    }).sort((a, b) => b.score - a.score);

    const withProb = getSoftmaxRanked(ranked, Math.max(0.02, profile.noise));
    const map = {};
    withProb.forEach(item => { map[item.idx] = item.prob; });
    return map;
}

// ==========================================
// 모델 초기화
// ==========================================
async function initModel() {
    const token = ++modelLoadToken;
    model = createAdaptiveModel();
    modelReady = true;
    setStartupStatus('기본 엔진으로 바로 시작할 수 있습니다. 이전 모델을 불러오는 중입니다.');
    updateUI();

    try {
        const loadedModel = await tf.loadLayersModel('localstorage://gomoku-adaptive-weights');
        if (token !== modelLoadToken) return;
        // 신규 CNN 구조 여부를 확인하여, 예전 모델이면 에러를 던져 초기화 유도
        if (loadedModel.inputs[0].shape[1] !== 225) throw new Error("Old model architecture");
        model = loadedModel;
        setStartupStatus('이전 모델까지 불러왔습니다.');
        updateUI();
    } catch (e) {
        if (token !== modelLoadToken) return;
        setStartupStatus('기본 엔진으로 실행 중입니다. 저장된 모델은 아직 불러오지 못했습니다.', true);
    }
}

// ==========================================
// 보드 평가 로직
// ==========================================
function countContinuousLength(b, x, y, dx, dy, p) {
    let count = 0, open = 0, nx = x + dx, ny = y + dy, blocked1 = false, blocked2 = false;
    while(nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE) {
        if(b[ny*SIZE + nx] === p) count++;
        else if(b[ny*SIZE + nx] === 0) { open++; break; }
        else { blocked1 = true; break; }
        nx += dx; ny += dy;
    }
    if (nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE) blocked1 = true;

    nx = x - dx; ny = y - dy;
    while(nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE) {
        if(b[ny*SIZE + nx] === p) count++;
        else if(b[ny*SIZE + nx] === 0) { open++; break; }
        else { blocked2 = true; break; }
        nx -= dx; ny -= dy;
    }
    if (nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE) blocked2 = true;

    return { count: count + 1, open, blocked: (blocked1 && blocked2) };
}

// getBaseScore is replaced by localized sliding windows in evaluateBoard
function evaluateBoard(b, targetPlayer) {
    const enemy = targetPlayer === 1 ? 2 : 1;
    let score = 0;

    const evaluateWindow = (pCount, eCount) => {
        if(eCount > 0) return 0;
        if(pCount === 5) return 10000000;
        if(pCount === 4) return 1000000;
        if(pCount === 3) return 10000;
        if(pCount === 2) return 100;
        if(pCount === 1) return 2;
        return 0;
    };

    // 중앙 선호도 부여
    for(let i=0; i<SIZE*SIZE; i++) {
        if(b[i] !== 0) {
            let x = i % SIZE, y = Math.floor(i / SIZE);
            let bias = Math.max(0, 80 - Math.sqrt((x-7)**2 + (y-7)**2) * 10);
            if(b[i] === targetPlayer) score += bias;
            else score -= bias * 2.5; 
        }
    }

    // 4방향 윈도우 스캔 (크기 5) - 끊어진 패턴(함정)까지 모두 탐지
    for(let y=0; y<SIZE; y++) {
        for(let x=0; x<SIZE; x++) {
            // 가로
            if(x <= SIZE-5) {
                let p1=0, e1=0, p2=0, e2=0;
                for(let k=0; k<5; k++) {
                    let v = b[y*SIZE + x + k];
                    if(v === targetPlayer) { p1++; e2++; }
                    else if(v === enemy) { e1++; p2++; }
                }
                score += evaluateWindow(p1, e1);
                score -= evaluateWindow(p2, e2) * 2.5;
            }
            // 세로
            if(y <= SIZE-5) {
                let p1=0, e1=0, p2=0, e2=0;
                for(let k=0; k<5; k++) {
                    let v = b[(y+k)*SIZE + x];
                    if(v === targetPlayer) { p1++; e2++; }
                    else if(v === enemy) { e1++; p2++; }
                }
                score += evaluateWindow(p1, e1);
                score -= evaluateWindow(p2, e2) * 2.5;
            }
            // 대각선 \
            if(x <= SIZE-5 && y <= SIZE-5) {
                let p1=0, e1=0, p2=0, e2=0;
                for(let k=0; k<5; k++) {
                    let v = b[(y+k)*SIZE + (x+k)];
                    if(v === targetPlayer) { p1++; e2++; }
                    else if(v === enemy) { e1++; p2++; }
                }
                score += evaluateWindow(p1, e1);
                score -= evaluateWindow(p2, e2) * 2.5;
            }
            // 대각선 /
            if(x >= 4 && y <= SIZE-5) {
                let p1=0, e1=0, p2=0, e2=0;
                for(let k=0; k<5; k++) {
                    let v = b[(y+k)*SIZE + (x-k)];
                    if(v === targetPlayer) { p1++; e2++; }
                    else if(v === enemy) { e1++; p2++; }
                }
                score += evaluateWindow(p1, e1);
                score -= evaluateWindow(p2, e2) * 2.5;
            }
        }
    }
    
    return score;
}

// 기존 UI 전용이었으나, 이제 봇의 핵심 엔진으로 승격된 깊은 수읽기 (Minimax) 절대 규범
function godModeEvaluate(b, depth, alpha, beta, maximizingPlayer) {
    let s = evaluateBoard(b, 2);
    if(depth === 0 || Math.abs(s) > PATTERNS.WIN*0.5) return s;
    let candidates = getCandidates(b);
    if(candidates.length === 0) return 0;

    if(maximizingPlayer) { 
        let maxV = -Infinity;
        for(let idx of candidates) {
            b[idx] = 2; let v = godModeEvaluate(b, depth - 1, alpha, beta, false); b[idx] = 0;
            maxV = Math.max(maxV, v); alpha = Math.max(alpha, v);
            if(beta <= alpha) break; 
        }
        return maxV;
    } else { 
        let minV = Infinity;
        for(let idx of candidates) {
            b[idx] = 1; let v = godModeEvaluate(b, depth - 1, alpha, beta, true); b[idx] = 0;
            minV = Math.min(minV, v); beta = Math.min(beta, v);
            if(beta <= alpha) break; 
        }
        return minV;
    }
}

function getCandidates(b) {
    let res = new Set();
    for(let i=0; i<SIZE*SIZE; i++) {
        if(b[i] !== 0) {
            let cx = i % SIZE, cy = Math.floor(i / SIZE);
            for(let dy=-1; dy<=1; dy++) {
                for(let dx=-1; dx<=1; dx++) {
                    let nx = cx+dx, ny = cy+dy;
                    if(nx>=0 && nx<SIZE && ny>=0 && ny<SIZE && b[ny*SIZE+nx] === 0) {
                        res.add(ny*SIZE+nx);
                    }
                }
            }
        }
    }
    let arr = Array.from(res);
    return arr.length ? arr : [112]; 
}

function checkWinOnBoard(b, x, y, p) {
    const dirs = [[1,0], [0,1], [1,1], [1,-1]];
    for (let [dx, dy] of dirs) {
        let count = 1;
        let nx = x + dx, ny = y + dy;
        while (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE && b[ny * SIZE + nx] === p) {
            count++;
            nx += dx;
            ny += dy;
        }
        nx = x - dx;
        ny = y - dy;
        while (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE && b[ny * SIZE + nx] === p) {
            count++;
            nx -= dx;
            ny -= dy;
        }
        if (count >= 5) return true;
    }
    return false;
}

function getImmediateWinningMoves(b, p, maxReturn = Infinity) {
    let wins = [];
    for (let i = 0; i < SIZE * SIZE; i++) {
        if (b[i] !== 0) continue;
        b[i] = p;
        const x = i % SIZE;
        const y = Math.floor(i / SIZE);
        if (checkWinOnBoard(b, x, y, p)) {
            wins.push(i);
            if (wins.length >= maxReturn) {
                b[i] = 0;
                break;
            }
        }
        b[i] = 0;
    }
    return wins;
}

function countNearbyStones(b, idx, p) {
    const x = idx % SIZE;
    const y = Math.floor(idx / SIZE);
    let c = 0;
    for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = x + dx;
            const ny = y + dy;
            if (nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE) continue;
            if (b[ny * SIZE + nx] === p) c++;
        }
    }
    return c;
}

function getUserPredictionMap(b) {
    const map = {};
    const empty = [];
    for (let i = 0; i < SIZE * SIZE; i++) if (b[i] === 0) empty.push(i);
    if (!empty.length) return map;

    const immediateWins = getImmediateWinningMoves(b, 1, 10);
    immediateWins.forEach(idx => { map[idx] = (map[idx] || 0) + 1.2; });

    let maxMoveCount = Math.max(1, ...playerModel.moveCount);
    let maxThreat = 1;
    Object.values(playerModel.threatCells).forEach(v => { maxThreat = Math.max(maxThreat, Number(v) || 0); });

    let maxTransition = 1;
    let transition = {};
    if (lastUserMoveIdx !== null && playerModel.transition[lastUserMoveIdx]) {
        transition = playerModel.transition[lastUserMoveIdx];
        Object.values(transition).forEach(v => { maxTransition = Math.max(maxTransition, Number(v) || 0); });
    }

    const lastX = lastUserMoveIdx === null ? 7 : lastUserMoveIdx % SIZE;
    const lastY = lastUserMoveIdx === null ? 7 : Math.floor(lastUserMoveIdx / SIZE);

    empty.forEach(idx => {
        const x = idx % SIZE;
        const y = Math.floor(idx / SIZE);
        const dist = Math.abs(x - lastX) + Math.abs(y - lastY);
        const localAggro = countNearbyStones(b, idx, 1) / 8;
        const moveFreq = (playerModel.moveCount[idx] || 0) / maxMoveCount;
        const threatFreq = (Number(playerModel.threatCells[idx]) || 0) / maxThreat;
        const transFreq = (Number(transition[idx]) || 0) / maxTransition;
        const recency = 1 / (1 + dist);

        const score = moveFreq * 0.25 + threatFreq * 0.35 + transFreq * 0.2 + localAggro * 0.15 + recency * 0.05;
        map[idx] = (map[idx] || 0) + score;
    });

    return map;
}

function updatePlayerModelAfterUserMove(x, y) {
    const idx = y * SIZE + x;
    playerModel.moveCount[idx] = (playerModel.moveCount[idx] || 0) + 1;

    if (lastUserMoveIdx !== null) {
        if (!playerModel.transition[lastUserMoveIdx]) playerModel.transition[lastUserMoveIdx] = {};
        const t = playerModel.transition[lastUserMoveIdx];
        t[idx] = (t[idx] || 0) + 1;
    }
    lastUserMoveIdx = idx;

    const userWinsSoon = getImmediateWinningMoves(board, 1, 6);
    userWinsSoon.forEach(wIdx => {
        playerModel.threatCells[wIdx] = (playerModel.threatCells[wIdx] || 0) + 1;
    });

    savePlayerModel();
}

async function getAdaptiveModifiersForCandidates(b, candidateIndices, player = 2) {
    let batchFeatures = [];
    
    for(let idx of candidateIndices) {
        b[idx] = player; 
        const feat = Array.from(b).map(v => {
            if (player === 2) return v === 1 ? 1 : (v === 2 ? -1 : 0);
            return v === 1 ? -1 : (v === 2 ? 1 : 0); // 유저 시점 역전 인식
        });
        batchFeatures.push(feat); 
        b[idx] = 0; 
    }

    return tf.tidy(() => {
        const tensor = tf.tensor2d(batchFeatures);
        const preds = model.predict(tensor);
        return preds.dataSync(); // 각 후보별 스칼라 배열 반환
    });
}

function buildProbabilityHeatmapFromMap(scoreMap, maxValue = 1) {
    return Object.entries(scoreMap).map(([idx, score]) => ({ idx: Number(idx), score: clamp(Number(score) / maxValue, 0, 1) }));
}

function clearDecisionTelemetry() {}

// ==========================================
// 인터랙션
// ==========================================
canvas.addEventListener('click', e => {
    if(isThinking) return;
    const rect = canvas.getBoundingClientRect();
    
    // 반응형 스케일 대응! canvas 내부 해상도 대비 현재 DOM 렌더 사이즈 비율
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = Math.round(((e.clientX - rect.left) * scaleX - PAD - CELL/2) / CELL);
    const y = Math.round(((e.clientY - rect.top) * scaleY - PAD - CELL/2) / CELL);
    
    if(x >= 0 && x < SIZE && y >= 0 && y < SIZE && board[y*SIZE + x] === 0) userMove(x, y);
});

canvas.addEventListener('mousemove', e => {
    if(isThinking) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = Math.round(((e.clientX - rect.left) * scaleX - PAD - CELL/2) / CELL);
    const y = Math.round(((e.clientY - rect.top) * scaleY - PAD - CELL/2) / CELL);
    
    if(x >= 0 && x < SIZE && y >= 0 && y < SIZE && board[y*SIZE + x] === 0) {
        if(!hoverPos || hoverPos.x !== x || hoverPos.y !== y) {
            hoverPos = {x, y};
            renderBoard();
        }
    } else if(hoverPos !== null) {
        hoverPos = null;
        renderBoard();
    }
});

canvas.addEventListener('mouseout', () => {
    if(hoverPos !== null) {
        hoverPos = null;
        renderBoard();
    }
});

async function userMove(x, y) {
    hoverPos = null;
    makeMove(x, y, 1);
    updatePlayerModelAfterUserMove(x, y);
    let winState = checkWin(x, y, 1);
    if(winState) {
        drawWinLine(winState);
        await sleep(1000);
        return endGame('won'); 
    }
    if(board.every(v => v !== 0)) return endGame('draw'); 
    isThinking = true; updateProb(); updateUI();
    botMove(); // 비동기로 자체 실행
}

async function botMove() {
    pushLog(`수읽기 중...`, 'bot-sys');
    await sleep(500);

    const profile = getDifficultyProfile(stats.total);
    let cans = getCandidates(board);
    let scoredCans = [];
    const userPredictionMap = getUserPredictionMap(board);
    const forcedBlocks = new Set(getImmediateWinningMoves(board, 1, 10));

    // 1단계: 끊어진 윈도우까지 탐지하는 향상된 상태 평가
    for(let idx of cans) {
        board[idx] = 2; 
        let rawScore = evaluateBoard(board, 2); 
        board[idx] = 0;
        scoredCans.push({idx, rawScore});
    }
    
    // 상위 후보 추리기
    scoredCans.sort((a,b) => b.rawScore - a.rawScore);
    let topCans = scoredCans.slice(0, profile.topN);
    
    await sleep(300);

    // 2단계: 진행도에 따라 깊이를 올리는 Minimax
    for(let i=0; i<topCans.length; i++) {
        board[topCans[i].idx] = 2; 
        topCans[i].rawScore = godModeEvaluate(board, profile.depth, -Infinity, Infinity, false); 
        board[topCans[i].idx] = 0;
    }
    
    // 수읽기 결과로 다시 정렬 (수읽기로 발견한 치명적 함정 픽은 점수가 나락으로 감)
    topCans.sort((a,b) => b.rawScore - a.rawScore);

    await sleep(200);

    // 3단계: 상위 후보들에 대해 신경망(Layer 2) + 사용자 예측 가중치 결합
    let topIndices = topCans.map(c => c.idx);
    let mods = await getAdaptiveModifiersForCandidates(board, topIndices);
    
    let ranked = [];
    let originalBest = topCans[0].idx;

    for(let i=0; i<topCans.length; i++) {
        let rawScore = topCans[i].rawScore;
        let mod = mods[i]; 
        let idx = topCans[i].idx;
        let predictionScore = userPredictionMap[idx] || 0;
        
        const adaptiveContribution = mod * profile.adaptiveWeight;
        const predictionContribution = predictionScore * profile.predictionWeight;
        let blendedScore = rawScore + adaptiveContribution + predictionContribution;

        board[idx] = 2;
        const userImmediateAfter = getImmediateWinningMoves(board, 1, 3).length;
        board[idx] = 0;

        if (forcedBlocks.size > 0 && !forcedBlocks.has(idx)) {
            blendedScore -= 800000;
        }
        blendedScore -= userImmediateAfter * 350000;
        
        // 확정적인 승/패 방어는 건드리지 않음
        if (Math.abs(rawScore) > PATTERNS.WIN * 0.5) blendedScore = rawScore;

        ranked.push({
            idx,
            score: blendedScore,
            rawScore,
            adaptiveContribution,
            predictionContribution
        });
    }

    ranked.sort((a, b) => b.score - a.score);
    const rankedWithProb = getSoftmaxRanked(ranked, Math.max(0.02, profile.noise));
    let best = rankedWithProb[0]?.idx ?? -1;

    // 초반에는 약점이 보이도록 탐험 노이즈를 크게 두고, 몇 판 지나면 빠르게 줄임
    if (rankedWithProb.length > 2 && Math.random() < profile.noise) {
        best = sampleBySoftmax(rankedWithProb.slice(0, Math.min(6, rankedWithProb.length)), profile.noise);
    }

    clearDecisionTelemetry();
    
    if (best !== originalBest) {
        pushLog(`착수 위치를 수정했습니다.`, 'bot-nn');
    } else {
        pushLog(`(${best%SIZE}, ${Math.floor(best/SIZE)})`, 'bot-act');
    }

    await sleep(200);

    if(best !== -1) {
        makeMove(best % SIZE, Math.floor(best / SIZE), 2);
        let winState = checkWin(best % SIZE, Math.floor(best / SIZE), 2);
        if(winState) {
            drawWinLine(winState);
            await sleep(1000);
            endGame('lost'); 
        }
        else if(board.every(v => v !== 0)) endGame('draw');
    }
    
    isThinking = false; updateProb(); updateUI();
}

function makeMove(x, y, p) {
    board[y*SIZE+x] = p;
    gameHistory.push({idx: y*SIZE+x, x, y, player: p, time: Date.now(), boardState: [...board]});
    
    // 진행 상황 로컬 스토리지에 자동 저장 (새로고침 방어)
    localStorage.setItem('gomoku-ongoing-board', JSON.stringify(board));
    localStorage.setItem('gomoku-ongoing-history', JSON.stringify(gameHistory));
    localStorage.setItem('gomoku-ongoing-prob', JSON.stringify(winProbTrace));

    recalculateHeatmap();
    renderBoard();
}

function checkWin(x, y, p) {
    const dirs = [[1,0], [0,1], [1,1], [1,-1]];
    for(let [dx, dy] of dirs) {
        let count = 1;
        let nx = x + dx, ny = y + dy;
        let end1 = {x, y}, end2 = {x, y};
        while(nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE && board[ny*SIZE + nx] === p) {
            count++; end1 = {x: nx, y: ny};
            nx += dx; ny += dy;
        }
        nx = x - dx; ny = y - dy;
        while(nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE && board[ny*SIZE + nx] === p) {
            count++; end2 = {x: nx, y: ny};
            nx -= dx; ny -= dy;
        }
        if(count >= 5) return { end1, end2 };
    }
    return null;
}

function drawWinLine(winState) {
    if (!winState) return;
    const { end1, end2 } = winState;
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(PAD + end1.x * CELL + CELL/2, PAD + end1.y * CELL + CELL/2);
    ctx.lineTo(PAD + end2.x * CELL + CELL/2, PAD + end2.y * CELL + CELL/2);
    ctx.stroke();
}

function updateProb() {
    // evaluateBoard(board, 2) 반환값은 봇 관점의 종합 스코어(봇점수 - 적점수)
    const diff = evaluateBoard(board, 2);
    // 점수 스케일 확장에 맞춰 정규화 상수 증가
    const p = 1 / (1 + Math.exp(-diff / 500000)); 
    winProbTrace.push(p);
}



// ==========================================
// 종료, 학습 및 갱신
// ==========================================
async function endGame(res) {
    isThinking = true;
    if(res === 'won') stats.won++; else if(res === 'lost') stats.lost++; else stats.draws++;
    stats.total++; 
    localStorage.setItem('gomoku-stats', JSON.stringify(stats));
    
    // 게임 종료 시 저장된 진행 상황 초기화
    localStorage.removeItem('gomoku-ongoing-board');
    localStorage.removeItem('gomoku-ongoing-history');
    localStorage.removeItem('gomoku-ongoing-prob');

    const avgProb = winProbTrace.reduce((a,b)=>a+b, 0) / winProbTrace.length;
    winProbHistory.push(avgProb);
    localStorage.setItem('gomoku-winprob-history', JSON.stringify(winProbHistory.slice(-20)));

    playerModel.gamesAnalyzed = (playerModel.gamesAnalyzed || 0) + 1;
    if (res === 'won') {
        // 패배한 판의 사용자 공격 좌표를 강하게 저장해서 같은 공격 재등장 시 빠르게 방어한다.
        gameHistory.forEach(h => {
            if (h.player !== 1) return;
            const w = h.idx;
            playerModel.threatCells[w] = (playerModel.threatCells[w] || 0) + 3;
        });
    }
    savePlayerModel();
    
    document.getElementById('modal-overlay').style.display = 'flex';
    document.getElementById('modal-title').innerText = res === 'won' ? "당신의 승리!" : (res === 'lost' ? "봇의 승리!" : "무승부!");
    document.getElementById('modal-body').innerText = "기록을 저장하고 학습합니다.";
    
    if(res === 'won') {
        pushLog(`패배를 학습했습니다.`, 'bot-warn');
    } else if(res === 'lost') {
        pushLog(`승리했습니다.`, 'bot-act');
    } else {
        pushLog(`무승부입니다.`, 'bot-act');
    }
    pushLog(`학습 저장 중...`, 'bot-nn');

    const reward = res === 'lost' ? 1 : (res === 'won' ? -1 : 0);
    let xsData = [], ysData = [];
    
    // 8방향 증강(Augmentation) 헬퍼 함수
    const getAugmented = (flatBoard) => {
        let results = [];
        for(let flip=0; flip<2; flip++) {
            for(let rot=0; rot<4; rot++) {
                let nb = new Array(225);
                for(let r=0; r<SIZE; r++) {
                    for(let c=0; c<SIZE; c++) {
                        let nr = r, nc = c;
                        if(flip) nc = SIZE - 1 - c;
                        for(let k=0; k<rot; k++) {
                            let tmp = nr;
                            nr = nc;
                            nc = SIZE - 1 - tmp;
                        }
                        nb[nr * SIZE + nc] = flatBoard[r * SIZE + c];
                    }
                }
                results.push(nb);
            }
        }
        return results;
    };

    gameHistory.forEach((h, i) => {
        if(h.player === 2) {
            let timeFactor = (i + 1) / gameHistory.length; 
            const feat = h.boardState.map(v => v === 1 ? 1 : (v === 2 ? -1 : 0));
            
            // 승패를 결정지은 공간 구조를 8개의 각도로 모두 회전/반전시켜 일괄 훈련 데이터에 추가
            const augmentedFeats = getAugmented(feat);
            augmentedFeats.forEach(augFeat => {
                xsData.push(augFeat);
                ysData.push([reward * timeFactor]);
            });
        }
    });

    if(xsData.length > 0) {
        const xs = tf.tensor2d(xsData), ys = tf.tensor2d(ysData);
        try {
            const h = await model.fit(xs, ys, { epochs: 3, batchSize: 64 });
            let fLoss = h.history.loss[0];
            lossHistory.push(fLoss); 
            localStorage.setItem('gomoku-loss-history', JSON.stringify(lossHistory.slice(-200)));
            pushLog(`Loss ${fLoss.toFixed(4)}`, 'bot-sys');
        } catch(e) {} finally { xs.dispose(); ys.dispose(); }
    }
    
    await model.save('localstorage://gomoku-adaptive-weights'); 
    document.getElementById('modal-body').innerText = "학습 완료! 게임판을 초기화합니다.";
    updateUI();
}



// ==========================================
// 렌더링
// ==========================================
function renderBoard() {
    ctx.clearRect(0,0,CANVAS_SIZE,CANVAS_SIZE); 
    ctx.strokeStyle = '#8B6914'; ctx.lineWidth = 1;
    for(let i=0; i<SIZE; i++) { 
        let p = PAD+i*CELL+CELL/2; 
        ctx.beginPath(); ctx.moveTo(PAD+CELL/2,p); ctx.lineTo(CANVAS_SIZE-PAD-CELL/2,p); ctx.stroke(); 
        ctx.beginPath(); ctx.moveTo(p,PAD+CELL/2); ctx.lineTo(p,CANVAS_SIZE-PAD-CELL/2); ctx.stroke(); 
    }
    [[3,3],[3,11],[11,3],[11,11],[7,7]].forEach(([r,c]) => { 
        ctx.fillStyle = '#8B6914'; ctx.beginPath(); ctx.arc(PAD+c*CELL+CELL/2,PAD+r*CELL+CELL/2,3,0,7); ctx.fill(); 
    });
    
    board.forEach((v,i) => { 
        if(v===0) return; 
        let x = PAD+(i%SIZE)*CELL+CELL/2, y = PAD+Math.floor(i/SIZE)*CELL+CELL/2; 
        ctx.beginPath(); let g = ctx.createRadialGradient(x-3,y-3,1,x,y,15); 
        if(v===1) { g.addColorStop(0,'#444'); g.addColorStop(1,'#000'); } else { g.addColorStop(0,'#fff'); g.addColorStop(1,'#ddd'); } 
        ctx.fillStyle=g; ctx.shadowBlur=4; ctx.shadowOffsetY=2; ctx.shadowColor='rgba(0,0,0,0.3)'; 
        ctx.arc(x,y,15,0,7); ctx.fill(); ctx.shadowBlur=0; ctx.shadowOffsetY=0; 
        if(gameHistory.length && gameHistory[gameHistory.length-1].idx === i) {
            ctx.fillStyle=v===1?'white':'black'; ctx.beginPath(); ctx.arc(x,y,2.5,0,7); ctx.fill();
        } 
    });

    if(hoverPos && !isThinking) {
        let hx = PAD+hoverPos.x*CELL+CELL/2, hy = PAD+hoverPos.y*CELL+CELL/2; 
        ctx.fillStyle = 'rgba(0,0,0,0.3)';
        ctx.beginPath(); ctx.arc(hx,hy,15,0,7); ctx.fill();
    }

    if(viewMode > 0) renderHeatmap();
}

let currentHeatmapData = null;

async function recalculateHeatmap() {
    if(viewMode === 0) return;
    let emptyIdxs = [];
    for(let i=0; i<SIZE*SIZE; i++) if(board[i] === 0) emptyIdxs.push(i);
    
    if (emptyIdxs.length === 0) return;

    let cells = [];
    if (viewMode === 1) {
        let mods = await getAdaptiveModifiersForCandidates(board, emptyIdxs, 2);
        for(let i=0; i<emptyIdxs.length; i++) {
            cells.push({idx: emptyIdxs[i], score: mods[i]}); 
        }
    } else if (viewMode === 2) {
        const predictionMap = getUserPredictionMap(board);
        let maxPred = 0.001;
        Object.values(predictionMap).forEach(v => { maxPred = Math.max(maxPred, Number(v) || 0); });
        for (let i = 0; i < emptyIdxs.length; i++) {
            const idx = emptyIdxs[i];
            const p = (predictionMap[idx] || 0) / maxPred;
            board[idx] = 1;
            const userEval = evaluateBoard(board, 1);
            board[idx] = 0;
            const blunderRisk = clamp((-userEval) / 200000, 0, 1);
            cells.push({ idx, score: clamp(p - blunderRisk * 0.8, -1, 1) });
        }
    } else {
        const botMap = await getBotCandidateHeatmap(board);
        let maxProb = 0.001;
        Object.values(botMap).forEach(v => { maxProb = Math.max(maxProb, Number(v) || 0); });
        for (let i = 0; i < emptyIdxs.length; i++) {
            const idx = emptyIdxs[i];
            const p = (botMap[idx] || 0) / maxProb;
            cells.push({ idx, score: clamp(p, 0, 1) });
        }
    }

    cells.sort((a,b) => b.score - a.score);
    currentHeatmapData = cells;
    renderBoard(); // 비동기이므로 끝난 후 다시 렌더해줌
}

function renderHeatmap() {
    if(!currentHeatmapData) return;
    let cells = currentHeatmapData;

    let sumPos = 0, countPos = 0;
    let sumNeg = 0, countNeg = 0;
    cells.forEach(c => {
        if(c.score > 0) { sumPos += c.score; countPos++; }
        else if(c.score < 0) { sumNeg += Math.abs(c.score); countNeg++; }
    });
    let avgPos = countPos ? Math.max(sumPos / countPos, 0.15) : 0;
    let avgNeg = countNeg ? Math.max(sumNeg / countNeg, 0.15) : 0;

    cells.forEach((c) => {
        let {idx: i, score: s} = c;
        if(viewMode === 3) {
            if(s < 0.04) return;
        } else {
            if(s > 0 && s < avgPos) return;
            if(s < 0 && Math.abs(s) < avgNeg) return;
            if(Math.abs(s) < 0.1) return;
        }
        
        const x = PAD + (i%SIZE)*CELL, y = PAD + Math.floor(i/SIZE)*CELL;
        
        if (viewMode === 1) {
            if (s > 0) ctx.fillStyle = `rgba(16, 185, 129, ${s * 0.8})`; 
            else ctx.fillStyle = `rgba(168, 85, 247, ${Math.abs(s) * 0.8})`; 
        } else if (viewMode === 2) {
            if (s > 0) ctx.fillStyle = `rgba(239, 68, 68, ${s * 0.8})`; 
            else ctx.fillStyle = `rgba(59, 130, 246, ${Math.abs(s) * 0.8})`; 
        } else {
            ctx.fillStyle = `rgba(239, 68, 68, ${0.18 + s * 0.82})`;
        }
        
        ctx.fillRect(x+1, y+1, CELL-2, CELL-2);
        
        if((viewMode === 3 && s > 0.12) || (viewMode !== 3 && Math.abs(s) > 0.3)) {
            ctx.fillStyle = 'white';
            ctx.font = '10px Inter';
            ctx.textAlign = 'center';
            let pct = Math.round(s * 100);
            ctx.fillText((pct > 0 ? '+' : '') + pct + '%', x+CELL/2, y+CELL/2 + 3);
        }
    });
}

function updateUI() {
    document.getElementById('game-counter').innerText = `${stats.total + 1}번째 판`;
    const lv = Math.min(10, Math.floor(stats.total/5)+1); const b = document.getElementById('bot-level');
    b.innerText = `Lv ${lv}: ${['','초보','적응 중','중수','숙련자','전문가','알파오목'][Math.min(6, Math.floor(lv/2)+1)]}`;
    b.className = `level-badge ${lv<4?'level-low':lv<7?'level-mid':'level-high'}`;
    document.getElementById('history-text').innerText = `총 ${stats.total}판 중 ${stats.won}승 ${stats.lost}패 ${stats.draws}무`;
    const p = winProbTrace[winProbTrace.length-1] || 0.5;
    document.getElementById('prob-user').style.width = (1-p)*100 + '%'; document.getElementById('prob-user').innerText = `당신 ${Math.round((1-p)*100)}%`;
    document.getElementById('prob-bot').innerText = `봇 ${Math.round(p*100)}%`;
    renderCharts();
}

function renderCharts() {
    const m = document.getElementById('momentum-chart'); 
    if (m) {
        let d = `M 0 ${80-(winProbTrace[0]*80)}`;
        for(let i=1; i<winProbTrace.length; i++) d += ` L ${(i/(winProbTrace.length-1))*320} ${80-(winProbTrace[i]*80)}`;
        m.innerHTML = `<svg viewBox="-5 0 330 80" class="chart-svg"><line x1="0" y1="40" x2="320" y2="40" stroke="#ddd" stroke-dasharray="4"/><path d="${d}" fill="none" stroke="${winProbTrace.at(-1)>0.5?'#ef4444':'#3b82f6'}" stroke-width="2.5"/></svg>`;
    }

    const l = document.getElementById('loss-chart'); 
    if(l && lossHistory.length) { 
        let ld = `M 0 80`; let maxL = Math.max(...lossHistory, 0.05); let slice = lossHistory.slice(-200);
        slice.forEach((v,i) => ld += ` L ${(i/Math.max(1, slice.length-1))*320} ${80-(v/maxL)*70}`);
        l.innerHTML = `<svg viewBox="-5 0 330 80" class="chart-svg"><path d="${ld}" fill="none" stroke="#ef4444" stroke-width="2"/></svg>`; 
        document.getElementById('stat-loss').innerText = lossHistory.at(-1).toFixed(4); 
    }
}

// ==========================================
// Export / Import 로직
// ==========================================

function exportMemory() {
    let memoryObject = {};
    for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        // 오목봇 관련 키와 TensorFlow.js 모델 키 추출
        if (key.startsWith('gomoku-') || key.startsWith('tensorflowjs_models/gomoku-adaptive-weights/')) {
            memoryObject[key] = localStorage.getItem(key);
        }
    }
    
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(memoryObject));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "omokbot_memory.json");
    document.body.appendChild(downloadAnchorNode); 
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
}

function importMemory(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const memoryObject = JSON.parse(e.target.result);
            
            // 기존 메모리 초기화 방지: 유효성 검사
            let isValid = false;
            let importedKeys = 0;
            
            // 로컬 스토리지 클리어 및 삽입
            localStorage.clear();
            for (const key in memoryObject) {
                if (key.startsWith('gomoku-') || key.startsWith('tensorflowjs_models/')) {
                    localStorage.setItem(key, memoryObject[key]);
                    isValid = true;
                    importedKeys++;
                }
            }

            if(isValid) {
                alert(`성공적으로 이어하기 데이터를 불러왔습니다! (${importedKeys}개의 세이브 파일)\n새로고침 됩니다.`);
                location.reload();
            } else {
                alert("적합한 오목봇 세이브 파일이 아닙니다.");
            }
        } catch (error) {
            alert("세이브 파일을 읽는 중 오류가 발생했습니다: " + error.message);
        }
        
        // input 초기화 (같은 파일 재업로드 허용 위해)
        event.target.value = '';
    };
    reader.readAsText(file);
}

// ==========================================
// 버튼 및 UI 이벤트 등록
// ==========================================
document.getElementById('btn-play-again').onclick = () => { 
    board = Array(SIZE * SIZE).fill(0); 
    gameHistory = []; 
    winProbTrace = [0.5]; 
    lastUserMoveIdx = null;
    
    // 모달 및 보드 상태 원복
    const overlay = document.getElementById('modal-overlay');
    overlay.style.display = 'none'; 
    overlay.style.background = '';
    const card = document.querySelector('#modal-overlay .modal-card');
    card.style.cssText = ''; 
    document.getElementById('btn-view-board').style.display = 'inline-block';
    
    isThinking = false; 
    recalculateHeatmap(); 
    renderBoard(); 
    updateUI(); 
};

if(document.getElementById('btn-view-board')) {
    document.getElementById('btn-view-board').onclick = () => {
        const overlay = document.getElementById('modal-overlay');
        overlay.style.background = 'transparent';
        const card = document.querySelector('#modal-overlay .modal-card');
        card.style.position = 'absolute';
        card.style.bottom = '20px';
        card.style.right = '20px';
        card.style.transform = 'scale(0.8)';
        card.style.transformOrigin = 'bottom right';
        document.getElementById('btn-view-board').style.display = 'none';
    };
}
document.getElementById('btn-toggle-heatmap').onclick = () => { 
    viewMode = (viewMode + 1) % 4;
    document.getElementById('btn-toggle-heatmap').innerText = VIEW_MODES[viewMode]; 
    document.getElementById('heatmap-legend').style.display = viewMode > 0 ? 'block' : 'none';
    
    if (viewMode === 1) {
        document.getElementById('legend-title').innerText = "AI 뇌파 분석 맵 (봇 관점)";
        document.getElementById('legend-pos-color').style.color = '#10b981';
        document.getElementById('legend-pos-color').innerText = "● 초록색(+)";
        document.getElementById('legend-pos-text').innerText = ": AI가 전략적으로 유리하다고 보는 자리";
        document.getElementById('legend-neg-color').style.color = '#a855f7';
        document.getElementById('legend-neg-color').innerText = "● 보라색(-)";
        document.getElementById('legend-neg-text').innerText = ": 과거 함정 경험으로 인해 회피하는 자리";
    } else if (viewMode === 2) {
        document.getElementById('legend-title').innerText = "사용자 수 확률 예측 맵 (실계산)";
        document.getElementById('legend-pos-color').style.color = '#ef4444';
        document.getElementById('legend-pos-color').innerText = "● 빨간색(+)";
        document.getElementById('legend-pos-text').innerText = ": 전이 패턴+위협도+국면을 합친 사용자 다음 수 확률이 높은 자리";
        document.getElementById('legend-neg-color').style.color = '#3b82f6';
        document.getElementById('legend-neg-color').innerText = "● 파란색(-)";
        document.getElementById('legend-neg-text').innerText = ": 사용자가 둘 확률은 있으나 즉시 악수로 분류된 자리";
    } else if (viewMode === 3) {
        document.getElementById('legend-title').innerText = "봇 후보 확률 맵 (실계산)";
        document.getElementById('legend-pos-color').style.color = '#ef4444';
        document.getElementById('legend-pos-color').innerText = "● 빨간색(+)")
        document.getElementById('legend-pos-text').innerText = ": 봇이 실제로 둘 확률이 높은 후보";
        document.getElementById('legend-neg-color').style.color = '#f59e0b';
        document.getElementById('legend-neg-color').innerText = "● 주황색(-)";
        document.getElementById('legend-neg-text').innerText = ": 상대적으로 낮은 확률의 후보";
    }

    recalculateHeatmap(); 
    renderBoard(); 
};
document.getElementById('btn-reset').onclick = () => {
    if(confirm("모든 기억과 승률 데이터를 지우시겠습니까?")) {
        localStorage.clear();
        playerModel = createDefaultPlayerModel();
        clearDecisionTelemetry();
        board = Array(SIZE * SIZE).fill(0);
        gameHistory = [];
        winProbTrace = [0.5];
        lastUserMoveIdx = null;
        isThinking = true;
        document.getElementById('startup-overlay').style.display = 'flex';
        recalculateHeatmap();
        renderBoard();
        updateUI();
    }
};
document.getElementById('btn-export').onclick = () => { exportMemory(); };
document.getElementById('import-file').addEventListener('change', importMemory);

// 아코디언 이벤트 (사이드 패널 토글)
document.querySelectorAll('.section-title').forEach(title => {
    title.addEventListener('click', () => {
        if(window.innerWidth <= 1024) {  // 모바일/태블릿에서만 토글 동작
            title.parentElement.classList.toggle('collapsed');
        }
    });
});

// 스타트업 모달 이벤트
document.getElementById('btn-start-new').addEventListener('click', () => {
    document.getElementById('startup-overlay').style.display = 'none';
    isThinking = false; 
    if (!modelReady) {
        pushLog('모델을 불러오는 중입니다. 준비되면 바로 시작할 수 있습니다.', 'bot-sys');
    }
});
document.getElementById('btn-start-load').addEventListener('click', () => {
    document.getElementById('import-file').click();
});

// 최초 실행 및 초기 상태
isThinking = true; // 모달을 닫기 전까지는 착수 제한
setStartupStatus('모델을 불러오는 중입니다. 잠시만 기다려 주세요.');
initModel(); recalculateHeatmap(); renderBoard();
window.addEventListener('resize', () => {
    renderBoard();
});
