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
const VIEW_MODES = ['시각화: 끄기', 'AI 뇌구조 맵 보기', '유저 공격 예측 보기'];
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

const PATTERNS = { WIN: 10000000, OPEN_4: 1000000, BLOCKED_4: 100000, OPEN_3: 10000, BLOCKED_3: 5000, OPEN_2: 100 };

// ==========================================
// 모델 초기화
// ==========================================
async function initModel() {
    try {
        model = await tf.loadLayersModel('localstorage://gomoku-finite-weights');
        // 신규 CNN 구조 여부를 확인하여, 예전 모델이면 에러를 던져 초기화 유도
        if (model.inputs[0].shape[1] !== 225) throw new Error("Old model architecture");
    } catch (e) {
        model = tf.sequential({ layers: [
            tf.layers.reshape({ targetShape: [15, 15, 1], inputShape: [225] }),
            tf.layers.conv2d({ filters: 32, kernelSize: 5, padding: 'same', activation: 'relu' }),
            tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu' }),
            tf.layers.flatten(),
            tf.layers.dense({ units: 64, activation: 'relu' }),
            tf.layers.dropout({ rate: 0.2 }),
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
    const thinkingMsgs = [
        `어디가 좋을지 잠시 수읽기 중입니다... 🤔`,
        `음, 신중하게 포석을 구상하고 있어요. 🧐`,
        `당신의 다음 수를 예측해 보는 중이에요. 💭`
    ];
    pushLog(thinkingMsgs[Math.floor(Math.random()*thinkingMsgs.length)], 'bot-sys');
    await sleep(800);

    let cans = getCandidates(board);
    let scoredCans = [];
    // 1단계: 끊어진 윈도우까지 탐지하는 향상된 상태 평가
    for(let idx of cans) {
        board[idx] = 2; 
        let rawScore = evaluateBoard(board, 2); 
        board[idx] = 0;
        scoredCans.push({idx, rawScore});
    }
    
    // 상위 15개 후보 추리기
    scoredCans.sort((a,b) => b.rawScore - a.rawScore);
    let topCans = scoredCans.slice(0, 15);
    
    // -- 사용자가 아쉬워했던 '수읽기 절대 규범' 해금! --
    pushLog(`봉인해둔 '수읽기 절대 규범(Minimax)'을 해제합니다. 상대의 다음 수를 예측해볼게요... 👁️`, 'bot-think');
    await sleep(800);

    // 2단계: 2-ply 수읽기 (유저의 1수 꿰뚫기)
    for(let i=0; i<topCans.length; i++) {
        board[topCans[i].idx] = 2; 
        topCans[i].rawScore = godModeEvaluate(board, 2, -Infinity, Infinity, false); 
        board[topCans[i].idx] = 0;
    }
    
    // 수읽기 결과로 다시 정렬 (수읽기로 발견한 치명적 함정 픽은 점수가 나락으로 감)
    topCans.sort((a,b) => b.rawScore - a.rawScore);

    if (stats.total > 0) {
        pushLog(`수읽기 완료! 이제 오답 노트를 꺼내 미세한 패턴 보정(신경망)을 거칩니다. 🧠`, 'bot-think');
    } else {
        pushLog(`수읽기 완료! 아직 학습 데이터는 없지만 제 얕은 직감을 보태볼게요. 🧠`, 'bot-think');
    }
    await sleep(800);

    // 3단계: 상위 후보들에 대해 신경망(Layer 2) 일괄 예측
    let topIndices = topCans.map(c => c.idx);
    let mods = await getAdaptiveModifiersForCandidates(board, topIndices);
    
    let best = -1, maxS = -Infinity;
    let originalBest = topCans[0].idx;
    let chosenMod = 0;

    for(let i=0; i<topCans.length; i++) {
        let rawScore = topCans[i].rawScore;
        let mod = mods[i]; 
        
        // 신경망 개입: 휴리스틱 점수에 유의미한 보정치 합산 (스케일 상향 반영)
        let blendedScore = rawScore + (mod * 20000); 
        
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
            pushLog(`잠깐! 방금 수읽기로 정한 자리가 제 신경망 오답 노트의 함정 확률과 유사하네요. 😱 수정합니다.`, 'bot-nn');
        } else {
            pushLog(`본능적으로 그 자리는 조금 불길해 보이네요. 전략을 수정합니다. 🧐`, 'bot-nn');
        }
        await sleep(800);
        pushLog(`수읽기 대신 직감을 믿고 (${best%SIZE}, ${Math.floor(best/SIZE)}) 위치에 착수합니다. ✨`, 'bot-act');
    } else {
        pushLog(`논리적 수읽기 엔진과 신경망 직감이 일치합니다! (${best%SIZE}, ${Math.floor(best/SIZE)})에 착수! 😊`, 'bot-act');
    }

    await sleep(600);

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
    
    document.getElementById('modal-overlay').style.display = 'flex';
    document.getElementById('modal-title').innerText = res === 'won' ? "당신의 승리!" : (res === 'lost' ? "봇의 승리!" : "무승부!");
    document.getElementById('modal-body').innerText = "기록을 분석하여 신경망을 미세 조정하고 있습니다...";
    
    if(res === 'won') {
        const loseMsgs = [
            `아앗... 제가 지다니요 😭 확실히 복기해서 다음엔 안 당할 거예요! 📝`,
            `훌륭한 수였습니다! 제 빈틈을 정확히 파고드셨네요. 다음엔 다를 겁니다! 🔥`,
            `패배를 인정합니다. 하지만 이 패배는 제 신경망의 훌륭한 거름이 될 거예요! 🌱`
        ];
        pushLog(loseMsgs[Math.floor(Math.random()*loseMsgs.length)], 'bot-warn');
    } else if(res === 'lost') {
        const winMsgs = [
            `제가 이겼네요! 그동안 학습한 패턴 방어 데이터가 효과를 톡톡히 봤습니다 🚀`,
            `후후, 제 계산대로 흘러갔군요! 이 승리 패턴은 더 강하게 기억될 겁니다. 😎`,
            `오목봇의 승리입니다! 당신의 패턴을 거의 다 파악한 것 같네요. 🤖`
        ];
        pushLog(winMsgs[Math.floor(Math.random()*winMsgs.length)], 'bot-act');
    } else {
        pushLog(`치열한 접전 끝에 무승부군요! 다음엔 꼭 결판을 내요. 🤝`, 'bot-act');
    }
    pushLog(`이번 대국의 기보를 바탕으로 가중치를 재조정하고 있습니다... 🛠️`, 'bot-nn');

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
            pushLog(`CNN 8방향 시각 인지 학습 완료! (오차: ${fLoss.toFixed(4)}) 다음엔 각도를 비틀어도 안 속아요 😎`, 'bot-sys');
        } catch(e) {} finally { xs.dispose(); ys.dispose(); }
    }
    
    await model.save('localstorage://gomoku-finite-weights'); 
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
    
    let mods = await getAdaptiveModifiersForCandidates(board, emptyIdxs, viewMode === 1 ? 2 : 1);
    let cells = [];
    for(let i=0; i<emptyIdxs.length; i++) {
        cells.push({idx: emptyIdxs[i], score: mods[i]}); 
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
    // 시각적 피로도를 줄이기 위해 가중치 평균을 내서 평균 이상인 것만 보여줌. (최소 0.15 임계치 사용)
    let avgPos = countPos ? Math.max(sumPos / countPos, 0.15) : 0;
    let avgNeg = countNeg ? Math.max(sumNeg / countNeg, 0.15) : 0;

    cells.forEach((c) => {
        let {idx: i, score: s} = c;
        if(s > 0 && s < avgPos) return; // 양수 중 평균 이하 숨김
        if(s < 0 && Math.abs(s) < avgNeg) return; // 음수 중 평균 절대치 이하 숨김
        if(Math.abs(s) < 0.1) return; // 유의미도 낮음
        
        const x = PAD + (i%SIZE)*CELL, y = PAD + Math.floor(i/SIZE)*CELL;
        
        if (viewMode === 1) {
            if (s > 0) ctx.fillStyle = `rgba(16, 185, 129, ${s * 0.8})`; 
            else ctx.fillStyle = `rgba(168, 85, 247, ${Math.abs(s) * 0.8})`; 
        } else {
            if (s > 0) ctx.fillStyle = `rgba(239, 68, 68, ${s * 0.8})`; 
            else ctx.fillStyle = `rgba(59, 130, 246, ${Math.abs(s) * 0.8})`; 
        }
        
        ctx.fillRect(x+1, y+1, CELL-2, CELL-2);
        
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
document.getElementById('btn-play-again').onclick = () => { 
    board = Array(SIZE * SIZE).fill(0); 
    gameHistory = []; 
    winProbTrace = [0.5]; 
    
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
    viewMode = (viewMode + 1) % 3;
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
        document.getElementById('legend-title').innerText = "사용자 수 예측 맵 (대리 관점)";
        document.getElementById('legend-pos-color').style.color = '#ef4444';
        document.getElementById('legend-pos-color').innerText = "● 빨간색(+)";
        document.getElementById('legend-pos-text').innerText = ": AI가 사용자의 다음 공격수로 예측하는 자리";
        document.getElementById('legend-neg-color').style.color = '#3b82f6';
        document.getElementById('legend-neg-color').innerText = "● 파란색(-)";
        document.getElementById('legend-neg-text').innerText = ": 사용자가 두면 자멸하는 비효율적인 수";
    }

    recalculateHeatmap(); 
    renderBoard(); 
};
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
