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
let heatmapEnabled = false, isThinking = false, model = null, modelReady = false;
let hoverPos = null; 

const sleep = ms => new Promise(r => setTimeout(r, ms));

function pushLog(msg, type='bot-sys') {
    const c = document.getElementById('ai-console');
    const d = document.createElement('div');
    d.className = `chat-bubble ${type}`;
    d.innerHTML = msg;
    c.appendChild(d);
    c.scrollTop = c.scrollHeight;
}

const canvas = document.getElementById('board-canvas'), ctx = canvas.getContext('2d');
canvas.width = canvas.height = CANVAS_SIZE;

const PATTERNS = { WIN: 1000000, OPEN_4: 100000, BLOCKED_4: 10000, OPEN_3: 5000, BLOCKED_3: 500, OPEN_2: 200 };

// ==========================================
// 모델 초기화
// ==========================================
async function initModel() {
    try {
        model = await tf.loadLayersModel('localstorage://gomoku-adaptive-weights');
    } catch (e) {
        model = tf.sequential({ layers: [
            tf.layers.dense({ units: 128, activation: 'relu', inputShape: [229] }), 
            tf.layers.dropout({ rate: 0.1 }),
            tf.layers.dense({ units: 64, activation: 'relu' }),
            tf.layers.dense({ units: 1, activation: 'tanh' })
        ]});
        model.compile({ optimizer: tf.train.adam(stats.total < 5 ? 0.005 : 0.001), loss: 'meanSquaredError' });
    }
    modelReady = true;
    updateUI();
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

function getBaseScore(b, x, y, p) {
    let score = 0;
    const directions = [[1,0], [0,1], [1,1], [1,-1]];
    score += Math.max(0, 80 - Math.sqrt((x-7)**2 + (y-7)**2) * 10);
    for(let [dx, dy] of directions) {
        const { count, open, blocked } = countContinuousLength(b, x, y, dx, dy, p);
        if(count >= 5) score += PATTERNS.WIN;
        else if(count === 4) score += open === 2 ? PATTERNS.OPEN_4 : (!blocked ? PATTERNS.BLOCKED_4 : 0);
        else if(count === 3) score += open === 2 ? PATTERNS.OPEN_3 : (!blocked ? PATTERNS.BLOCKED_3 : 0);
        else if(count === 2) score += open === 2 ? PATTERNS.OPEN_2 : 0;
    }
    return score;
}

function evaluateBoard(b, targetPlayer) {
    let s = 0;
    for(let i=0; i<SIZE*SIZE; i++) {
        if(b[i] === targetPlayer) s += getBaseScore(b, i % SIZE, Math.floor(i / SIZE), targetPlayer);
        else if(b[i] !== 0) s -= getBaseScore(b, i % SIZE, Math.floor(i / SIZE), b[i]) * 1.2; 
    }
    return s;
}

// UI 승률, 히트맵 예측 전용 (오목봇 자신은 사용하지 못하는 '수읽기' 절대 규범)
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

async function getAdaptiveModifiersForCandidates(b, candidateIndices) {
    const meta = [gameHistory.length/225, stats.won/(stats.total||1), stats.total/50, 0.5];
    let batchFeatures = [];
    
    for(let idx of candidateIndices) {
        b[idx] = 2; // 현재 후보에 돌을 놓아본 가상 보드
        const feat = Array.from(b).map(v => v === 1 ? 1 : (v === 2 ? -1 : 0));
        batchFeatures.push(feat.concat(meta));
        b[idx] = 0; // 원복
    }

    return tf.tidy(() => {
        const tensor = tf.tensor2d(batchFeatures);
        const preds = model.predict(tensor);
        return preds.dataSync(); // 각 후보별 스칼라 배열 반환
    });
}

// ==========================================
// 인터랙션
// ==========================================
canvas.addEventListener('click', e => {
    if(isThinking || !modelReady) return;
    const rect = canvas.getBoundingClientRect();
    
    // 반응형 스케일 대응! canvas 내부 해상도 대비 현재 DOM 렌더 사이즈 비율
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = Math.round(((e.clientX - rect.left) * scaleX - PAD - CELL/2) / CELL);
    const y = Math.round(((e.clientY - rect.top) * scaleY - PAD - CELL/2) / CELL);
    
    if(x >= 0 && x < SIZE && y >= 0 && y < SIZE && board[y*SIZE + x] === 0) userMove(x, y);
});

canvas.addEventListener('mousemove', e => {
    if(isThinking || !modelReady) return;
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
    if(checkWin(x, y, 1)) return endGame('won'); 
    if(board.every(v => v !== 0)) return endGame('draw'); 
    isThinking = true; updateProb(); updateUI();
    botMove(); // 비동기로 자체 실행
}

async function botMove() {
    pushLog(`음... 어디에 두면 좋을지 생각 중이에요! 🤔`, 'bot-sys');
    await sleep(800);

    let cans = getCandidates(board);
    let scoredCans = [];
    // 1단계: 순수 상태 평가로 모든 후보 평가 (딥 서치 제거 - 트랩에 취약해짐)
    for(let idx of cans) {
        board[idx] = 2; 
        let rawScore = evaluateBoard(board, 2); 
        board[idx] = 0;
        scoredCans.push({idx, rawScore});
    }
    
    // 상위 15개 후보 추리기
    scoredCans.sort((a,b) => b.rawScore - a.rawScore);
    let topCans = scoredCans.slice(0, 15);
    
    if (stats.total > 0) {
        pushLog(`잠깐만요, 예전에 당신한테 당했던 함정이 있는지 제 기억(신경망)을 떠올려볼게요... 🧠`, 'bot-think');
    } else {
        pushLog(`아직 배운 건 없지만, 제 동물적인 직감(초기 신경망)에 의존해 볼게요... 🧠`, 'bot-think');
    }
    await sleep(1200);

    // 2단계: 상위 후보들에 대해 신경망(Layer 2) 일괄 예측
    let topIndices = topCans.map(c => c.idx);
    let mods = await getAdaptiveModifiersForCandidates(board, topIndices);
    
    let best = -1, maxS = -Infinity;
    let originalBest = topCans[0].idx;
    let chosenMod = 0;

    for(let i=0; i<topCans.length; i++) {
        let rawScore = topCans[i].rawScore;
        let mod = mods[i]; 
        
        // 신경망 개입: 휴리스틱 점수에 유의미한 보정치 합산
        let blendedScore = rawScore + (mod * 8000); 
        
        // 확정적인 승/패 방어는 건드리지 않음
        if (Math.abs(rawScore) > PATTERNS.WIN * 0.5) blendedScore = rawScore;
        
        if(blendedScore > maxS) { 
            maxS = blendedScore; 
            best = topCans[i].idx; 
            chosenMod = mod;
        }
    }
    
    if (best !== originalBest) {
        if (stats.total > 0) {
            pushLog(`앗! 방금 그곳은 예전에 당했던 함정 패턴과 비슷해요! 😱 본능을 거스르고 피해야겠어요.`, 'bot-nn');
        } else {
            pushLog(`왠지 모르게 직감적으로 위험해 보이네요! 🧐 본능을 거스르고 다른 곳을 찾아볼게요.`, 'bot-nn');
        }
        await sleep(800);
        pushLog(`대신 상대적으로 안전해 보이는 (${best%SIZE}, ${Math.floor(best/SIZE)})에 둘게요! ✨`, 'bot-act');
    } else {
        pushLog(`별다른 위험은 보이지 않네요. 예정대로 (${best%SIZE}, ${Math.floor(best/SIZE)})에 착수! 😊`, 'bot-act');
    }

    await sleep(600);

    if(best !== -1) {
        makeMove(best % SIZE, Math.floor(best / SIZE), 2);
        if(checkWin(best % SIZE, Math.floor(best / SIZE), 2)) endGame('lost'); 
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
        const { count } = countContinuousLength(board, x, y, dx, dy, p);
        if(count >= 5) return true;
    }
    return false;
}

function updateProb() {
    // UI 로직: 깊은 수읽기를 사용하여 현재 보드의 '진짜' 승률 계산
    const s = godModeEvaluate(board, 1, -Infinity, Infinity, true); 
    const p = 1 / (1 + Math.exp(-s / 100000)); 
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
    
    document.getElementById('modal-overlay').style.display = 'flex';
    document.getElementById('modal-title').innerText = res === 'won' ? "당신의 승리!" : (res === 'lost' ? "봇의 승리!" : "무승부!");
    document.getElementById('modal-body').innerText = "기록을 분석하여 신경망을 미세 조정하고 있습니다...";
    
    if(res === 'won') {
        pushLog(`아앗... 제가 짓다니요 😭 확실히 복기해서 다음번엔 똑같은 함정에 안 당할 거예요! 📝`, 'bot-warn');
    } else if(res === 'lost') {
        pushLog(`제가 이겼네요! 제 데이터가 제대로 작동하나 봐요 🚀 패턴을 강화합니다!`, 'bot-act');
    }
    pushLog(`기보 데이터를 바탕으로 인공지능 뇌구조를 재공사합니다... 🛠️`, 'bot-nn');

    const reward = res === 'lost' ? 1 : (res === 'won' ? -1 : 0);
    let xsData = [], ysData = [];
    gameHistory.forEach((h, i) => {
        if(h.player === 2) {
            let timeFactor = (i + 1) / gameHistory.length; 
            const feat = h.boardState.map(v => v === 1 ? 1 : (v === 2 ? -1 : 0));
            const meta = [i/225, stats.won/stats.total, stats.total/50, 0.5];
            xsData.push(feat.concat(meta)); ysData.push([reward * timeFactor]);
        }
    });

    if(xsData.length > 0) {
        const xs = tf.tensor2d(xsData), ys = tf.tensor2d(ysData);
        try {
            const h = await model.fit(xs, ys, { epochs: 3, batchSize: 32 });
            let fLoss = h.history.loss[0];
            lossHistory.push(fLoss); 
            localStorage.setItem('gomoku-loss-history', JSON.stringify(lossHistory.slice(-200)));
            pushLog(`학습 완료! (오차율: ${fLoss.toFixed(4)}) 다음 판에서는 더 똑똑해질 거예요 😎`, 'bot-sys');
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

    if(heatmapEnabled) renderHeatmap();
}

let currentHeatmapData = null;

async function recalculateHeatmap() {
    if(!heatmapEnabled) return;
    let emptyIdxs = [];
    for(let i=0; i<SIZE*SIZE; i++) if(board[i] === 0) emptyIdxs.push(i);
    
    if (emptyIdxs.length === 0) return;
    
    // 비동기 일괄 예측
    let mods = await getAdaptiveModifiersForCandidates(board, emptyIdxs);
    let cells = [];
    for(let i=0; i<emptyIdxs.length; i++) {
        // mod 값은 -1.0 ~ 1.0. 
        // 봇에게 좋은 자리(추천)는 +, 피해야 할 곳(함정)은 -
        cells.push({idx: emptyIdxs[i], score: mods[i]}); 
    }
    cells.sort((a,b) => b.score - a.score);
    currentHeatmapData = cells;
    renderBoard(); // 비동기이므로 끝난 후 다시 렌더해줌
}

function renderHeatmap() {
    if(!currentHeatmapData) return;
    let cells = currentHeatmapData;

    cells.forEach((c) => {
        let {idx: i, score: s} = c;
        if(Math.abs(s) < 0.1) return; // 무의미한 가중치는 렌더링 생략
        
        const x = PAD + (i%SIZE)*CELL, y = PAD + Math.floor(i/SIZE)*CELL;
        
        // 긍정적 보상(녹색 계열), 부정적 보상(보라/마젠타 계열)
        if (s > 0) {
            ctx.fillStyle = `rgba(16, 185, 129, ${s * 0.7})`; // Green
        } else {
            ctx.fillStyle = `rgba(168, 85, 247, ${Math.abs(s) * 0.7})`; // Purple
        }
        ctx.fillRect(x+1, y+1, CELL-2, CELL-2);
        
        // 유의미한 수치 표시
        if(Math.abs(s) > 0.3) {
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
document.getElementById('btn-play-again').onclick = () => { board = Array(SIZE * SIZE).fill(0); gameHistory = []; winProbTrace = [0.5]; document.getElementById('modal-overlay').style.display = 'none'; isThinking = false; recalculateHeatmap(); renderBoard(); updateUI(); };
document.getElementById('btn-toggle-heatmap').onclick = () => { heatmapEnabled = !heatmapEnabled; document.getElementById('btn-toggle-heatmap').innerText = `예측 맵: ${heatmapEnabled ? '켜짐' : '끄기'}`; recalculateHeatmap(); renderBoard(); };
document.getElementById('btn-reset').onclick = () => { if(confirm("모든 기억과 승률 데이터를 지우시겠습니까?")) { localStorage.clear(); location.reload(); } };
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
    if(!modelReady) { alert('AI 모델을 불러오는 중입니다. 잠시만 기다려주세요!'); return; }
    document.getElementById('startup-overlay').style.display = 'none';
    isThinking = false; 
});
document.getElementById('btn-start-load').addEventListener('click', () => {
    document.getElementById('import-file').click();
});

// 최초 실행 및 초기 상태
isThinking = true; // 모달을 닫기 전까지는 착수 제한
initModel(); recalculateHeatmap(); renderBoard();
window.addEventListener('resize', () => {
    renderBoard();
});
