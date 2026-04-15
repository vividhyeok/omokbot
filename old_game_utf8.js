// ==========================================
// ?ㅼ젙 諛??곹깭
// ==========================================
const SIZE = 15, CELL = 35, PAD = 20, CANVAS_SIZE = SIZE * CELL + PAD * 2;
let board = JSON.parse(localStorage.getItem('gomoku-ongoing-board')) || Array(SIZE * SIZE).fill(0);
let gameHistory = JSON.parse(localStorage.getItem('gomoku-ongoing-history')) || []; 
let winProbTrace = JSON.parse(localStorage.getItem('gomoku-ongoing-prob')) || [0.5];

// LocalStorage 媛??뚯떛
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
// 紐⑤뜽 珥덇린??// ==========================================
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
// 蹂대뱶 ?됯? 濡쒖쭅
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

// UI ?밸쪧, ?덊듃留??덉륫 ?꾩슜 (?ㅻぉ遊??먯떊? ?ъ슜?섏? 紐삵븯??'?섏씫湲? ?덈? 洹쒕쾾)
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
        b[idx] = 2; // ?꾩옱 ?꾨낫???뚯쓣 ?볦븘蹂?媛??蹂대뱶
        const feat = Array.from(b).map(v => v === 1 ? 1 : (v === 2 ? -1 : 0));
        batchFeatures.push(feat.concat(meta));
        b[idx] = 0; // ?먮났
    }

    return tf.tidy(() => {
        const tensor = tf.tensor2d(batchFeatures);
        const preds = model.predict(tensor);
        return preds.dataSync(); // 媛??꾨낫蹂??ㅼ뭡??諛곗뿴 諛섑솚
    });
}

// ==========================================
// ?명꽣?숈뀡
// ==========================================
canvas.addEventListener('click', e => {
    if(isThinking || !modelReady) return;
    const rect = canvas.getBoundingClientRect();
    
    // 諛섏쓳???ㅼ?????? canvas ?대? ?댁긽???鍮??꾩옱 DOM ?뚮뜑 ?ъ씠利?鍮꾩쑉
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
    botMove(); // 鍮꾨룞湲곕줈 ?먯껜 ?ㅽ뻾
}

async function botMove() {
    pushLog(`??.. ?대뵒???먮㈃ 醫뗭쓣吏 ?앷컖 以묒씠?먯슂! ?쨺`, 'bot-sys');
    await sleep(800);

    let cans = getCandidates(board);
    let scoredCans = [];
    // 1?④퀎: ?쒖닔 ?곹깭 ?됯?濡?紐⑤뱺 ?꾨낫 ?됯? (???쒖튂 ?쒓굅 - ?몃옪??痍⑥빟?댁쭚)
    for(let idx of cans) {
        board[idx] = 2; 
        let rawScore = evaluateBoard(board, 2); 
        board[idx] = 0;
        scoredCans.push({idx, rawScore});
    }
    
    // ?곸쐞 15媛??꾨낫 異붾━湲?    scoredCans.sort((a,b) => b.rawScore - a.rawScore);
    let topCans = scoredCans.slice(0, 15);
    
    if (stats.total > 0) {
        pushLog(`?좉퉸留뚯슂, ?덉쟾???뱀떊?쒗뀒 ?뱁뻽???⑥젙???덈뒗吏 ??湲곗뼲(?좉꼍留????좎삱?ㅻ낵寃뚯슂... ?쭬`, 'bot-think');
    } else {
        pushLog(`?꾩쭅 諛곗슫 嫄??놁?留? ???숇Ъ?곸씤 吏곴컧(珥덇린 ?좉꼍留????섏〈??蹂쇨쾶??.. ?쭬`, 'bot-think');
    }
    await sleep(1200);

    // 2?④퀎: ?곸쐞 ?꾨낫?ㅼ뿉 ????좉꼍留?Layer 2) ?쇨큵 ?덉륫
    let topIndices = topCans.map(c => c.idx);
    let mods = await getAdaptiveModifiersForCandidates(board, topIndices);
    
    let best = -1, maxS = -Infinity;
    let originalBest = topCans[0].idx;
    let chosenMod = 0;

    for(let i=0; i<topCans.length; i++) {
        let rawScore = topCans[i].rawScore;
        let mod = mods[i]; 
        
        // ?좉꼍留?媛쒖엯: ?대━?ㅽ떛 ?먯닔???좎쓽誘명븳 蹂댁젙移??⑹궛
        let blendedScore = rawScore + (mod * 8000); 
        
        // ?뺤젙?곸씤 ????諛⑹뼱??嫄대뱶由ъ? ?딆쓬
        if (Math.abs(rawScore) > PATTERNS.WIN * 0.5) blendedScore = rawScore;
        
        if(blendedScore > maxS) { 
            maxS = blendedScore; 
            best = topCans[i].idx; 
            chosenMod = mod;
        }
    }
    
    if (best !== originalBest) {
        if (stats.total > 0) {
            pushLog(`?? 諛⑷툑 洹멸납? ?덉쟾???뱁뻽???⑥젙 ?⑦꽩怨?鍮꾩듂?댁슂! ?삺 蹂몃뒫??嫄곗뒪瑜닿퀬 ?쇳빐?쇨쿋?댁슂.`, 'bot-nn');
        } else {
            pushLog(`?좎? 紐⑤Ⅴ寃?吏곴컧?곸쑝濡??꾪뿕??蹂댁씠?ㅼ슂! ?쭚 蹂몃뒫??嫄곗뒪瑜닿퀬 ?ㅻⅨ 怨녹쓣 李얠븘蹂쇨쾶??`, 'bot-nn');
        }
        await sleep(800);
        pushLog(`????곷??곸쑝濡??덉쟾??蹂댁씠??(${best%SIZE}, ${Math.floor(best/SIZE)})???섍쾶?? ??, 'bot-act');
    } else {
        pushLog(`蹂꾨떎瑜??꾪뿕? 蹂댁씠吏 ?딅꽕?? ?덉젙?濡?(${best%SIZE}, ${Math.floor(best/SIZE)})??李⑹닔! ?삃`, 'bot-act');
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
    
    // 吏꾪뻾 ?곹솴 濡쒖뺄 ?ㅽ넗由ъ????먮룞 ???(?덈줈怨좎묠 諛⑹뼱)
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
    let botScore = 0, userScore = 0;
    for(let i=0; i<SIZE*SIZE; i++) {
        if(board[i]===2) botScore += getBaseScore(board, i%SIZE, Math.floor(i/SIZE), 2);
        else if(board[i]===1) userScore += getBaseScore(board, i%SIZE, Math.floor(i/SIZE), 1);
    }
    // ?곷????밸━ 吏곸쟾(OPEN_4 ???????먯닔 寃⑹감媛 ?꾩껌?섍쾶 踰뚯뼱吏?
    const diff = botScore - userScore;
    const p = 1 / (1 + Math.exp(-diff / 30000)); 
    winProbTrace.push(p);
}



// ==========================================
// 醫낅즺, ?숈뒿 諛?媛깆떊
// ==========================================
async function endGame(res) {
    isThinking = true;
    if(res === 'won') stats.won++; else if(res === 'lost') stats.lost++; else stats.draws++;
    stats.total++; 
    localStorage.setItem('gomoku-stats', JSON.stringify(stats));
    
    // 寃뚯엫 醫낅즺 ????λ맂 吏꾪뻾 ?곹솴 珥덇린??    localStorage.removeItem('gomoku-ongoing-board');
    localStorage.removeItem('gomoku-ongoing-history');
    localStorage.removeItem('gomoku-ongoing-prob');

    const avgProb = winProbTrace.reduce((a,b)=>a+b, 0) / winProbTrace.length;
    winProbHistory.push(avgProb);
    localStorage.setItem('gomoku-winprob-history', JSON.stringify(winProbHistory.slice(-20)));
    
    document.getElementById('modal-overlay').style.display = 'flex';
    document.getElementById('modal-title').innerText = res === 'won' ? "?뱀떊???밸━!" : (res === 'lost' ? "遊뉗쓽 ?밸━!" : "臾댁듅遺!");
    document.getElementById('modal-body').innerText = "湲곕줉??遺꾩꽍?섏뿬 ?좉꼍留앹쓣 誘몄꽭 議곗젙?섍퀬 ?덉뒿?덈떎...";
    
    if(res === 'won') {
        pushLog(`?꾩븮... ?쒓? 吏?ㅻ땲???삲 ?뺤떎??蹂듦린?댁꽌 ?ㅼ쓬踰덉뿏 ?묎컳? ?⑥젙?????뱁븷 嫄곗삁?? ?뱷`, 'bot-warn');
    } else if(res === 'lost') {
        pushLog(`?쒓? ?닿꼈?ㅼ슂! ???곗씠?곌? ?쒕?濡??묐룞?섎굹 遊먯슂 ?? ?⑦꽩??媛뺥솕?⑸땲??`, 'bot-act');
    }
    pushLog(`湲곕낫 ?곗씠?곕? 諛뷀깢?쇰줈 ?멸났吏???뚭뎄議곕? ?ш났?ы빀?덈떎... ?썱截?, 'bot-nn');

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
            pushLog(`?숈뒿 ?꾨즺! (?ㅼ감?? ${fLoss.toFixed(4)}) ?ㅼ쓬 ?먯뿉?쒕뒗 ???묐삊?댁쭏 嫄곗삁???삇`, 'bot-sys');
        } catch(e) {} finally { xs.dispose(); ys.dispose(); }
    }
    
    await model.save('localstorage://gomoku-adaptive-weights'); 
    document.getElementById('modal-body').innerText = "?숈뒿 ?꾨즺! 寃뚯엫?먯쓣 珥덇린?뷀빀?덈떎.";
    updateUI();
}



// ==========================================
// ?뚮뜑留?// ==========================================
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
    
    // 鍮꾨룞湲??쇨큵 ?덉륫
    let mods = await getAdaptiveModifiersForCandidates(board, emptyIdxs);
    let cells = [];
    for(let i=0; i<emptyIdxs.length; i++) {
        // mod 媛믪? -1.0 ~ 1.0. 
        // 遊뉗뿉寃?醫뗭? ?먮━(異붿쿇)??+, ?쇳빐????怨??⑥젙)? -
        cells.push({idx: emptyIdxs[i], score: mods[i]}); 
    }
    cells.sort((a,b) => b.score - a.score);
    currentHeatmapData = cells;
    renderBoard(); // 鍮꾨룞湲곗씠誘濡??앸궃 ???ㅼ떆 ?뚮뜑?댁쨲
}

function renderHeatmap() {
    if(!currentHeatmapData) return;
    let cells = currentHeatmapData;

    cells.forEach((c) => {
        let {idx: i, score: s} = c;
        if(Math.abs(s) < 0.1) return; // 臾댁쓽誘명븳 媛以묒튂???뚮뜑留??앸왂
        
        const x = PAD + (i%SIZE)*CELL, y = PAD + Math.floor(i/SIZE)*CELL;
        
        // 湲띿젙??蹂댁긽(?뱀깋 怨꾩뿴), 遺?뺤쟻 蹂댁긽(蹂대씪/留덉젨? 怨꾩뿴)
        if (s > 0) {
            ctx.fillStyle = `rgba(16, 185, 129, ${s * 0.7})`; // Green
        } else {
            ctx.fillStyle = `rgba(168, 85, 247, ${Math.abs(s) * 0.7})`; // Purple
        }
        ctx.fillRect(x+1, y+1, CELL-2, CELL-2);
        
        // ?좎쓽誘명븳 ?섏튂 ?쒖떆
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
    document.getElementById('game-counter').innerText = `${stats.total + 1}踰덉㎏ ??;
    const lv = Math.min(10, Math.floor(stats.total/5)+1); const b = document.getElementById('bot-level');
    b.innerText = `Lv ${lv}: ${['','珥덈낫','?곸쓳 以?,'以묒닔','?숇젴??,'?꾨Ц媛','?뚰뙆?ㅻぉ'][Math.min(6, Math.floor(lv/2)+1)]}`;
    b.className = `level-badge ${lv<4?'level-low':lv<7?'level-mid':'level-high'}`;
    document.getElementById('history-text').innerText = `珥?${stats.total}??以?${stats.won}??${stats.lost}??${stats.draws}臾?;
    const p = winProbTrace[winProbTrace.length-1] || 0.5;
    document.getElementById('prob-user').style.width = (1-p)*100 + '%'; document.getElementById('prob-user').innerText = `?뱀떊 ${Math.round((1-p)*100)}%`;
    document.getElementById('prob-bot').innerText = `遊?${Math.round(p*100)}%`;
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
// Export / Import 濡쒖쭅
// ==========================================

function exportMemory() {
    let memoryObject = {};
    for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        // ?ㅻぉ遊?愿???ㅼ? TensorFlow.js 紐⑤뜽 ??異붿텧
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
            
            // 湲곗〈 硫붾え由?珥덇린??諛⑹?: ?좏슚??寃??            let isValid = false;
            let importedKeys = 0;
            
            // 濡쒖뺄 ?ㅽ넗由ъ? ?대━??諛??쎌엯
            localStorage.clear();
            for (const key in memoryObject) {
                if (key.startsWith('gomoku-') || key.startsWith('tensorflowjs_models/')) {
                    localStorage.setItem(key, memoryObject[key]);
                    isValid = true;
                    importedKeys++;
                }
            }

            if(isValid) {
                alert(`?깃났?곸쑝濡??댁뼱?섍린 ?곗씠?곕? 遺덈윭?붿뒿?덈떎! (${importedKeys}媛쒖쓽 ?몄씠釉??뚯씪)\n?덈줈怨좎묠 ?⑸땲??`);
                location.reload();
            } else {
                alert("?곹빀???ㅻぉ遊??몄씠釉??뚯씪???꾨떃?덈떎.");
            }
        } catch (error) {
            alert("?몄씠釉??뚯씪???쎈뒗 以??ㅻ쪟媛 諛쒖깮?덉뒿?덈떎: " + error.message);
        }
        
        // input 珥덇린??(媛숈? ?뚯씪 ?ъ뾽濡쒕뱶 ?덉슜 ?꾪빐)
        event.target.value = '';
    };
    reader.readAsText(file);
}

// ==========================================
// 踰꾪듉 諛?UI ?대깽???깅줉
// ==========================================
document.getElementById('btn-play-again').onclick = () => { board = Array(SIZE * SIZE).fill(0); gameHistory = []; winProbTrace = [0.5]; document.getElementById('modal-overlay').style.display = 'none'; isThinking = false; recalculateHeatmap(); renderBoard(); updateUI(); };
document.getElementById('btn-toggle-heatmap').onclick = () => { 
    heatmapEnabled = !heatmapEnabled; 
    document.getElementById('btn-toggle-heatmap').innerText = `AI ?뚭뎄議?留? ${heatmapEnabled ? '耳쒖쭚' : '?꾧린'}`; 
    document.getElementById('heatmap-legend').style.display = heatmapEnabled ? 'block' : 'none';
    recalculateHeatmap(); 
    renderBoard(); 
};
document.getElementById('btn-reset').onclick = () => { if(confirm("紐⑤뱺 湲곗뼲怨??밸쪧 ?곗씠?곕? 吏?곗떆寃좎뒿?덇퉴?")) { localStorage.clear(); location.reload(); } };
document.getElementById('btn-export').onclick = () => { exportMemory(); };
document.getElementById('import-file').addEventListener('change', importMemory);

// ?꾩퐫?붿뼵 ?대깽??(?ъ씠???⑤꼸 ?좉?)
document.querySelectorAll('.section-title').forEach(title => {
    title.addEventListener('click', () => {
        if(window.innerWidth <= 1024) {  // 紐⑤컮???쒕툝由우뿉?쒕쭔 ?좉? ?숈옉
            title.parentElement.classList.toggle('collapsed');
        }
    });
});

// ?ㅽ??몄뾽 紐⑤떖 ?대깽??document.getElementById('btn-start-new').addEventListener('click', () => {
    if(!modelReady) { alert('AI 紐⑤뜽??遺덈윭?ㅻ뒗 以묒엯?덈떎. ?좎떆留?湲곕떎?ㅼ＜?몄슂!'); return; }
    document.getElementById('startup-overlay').style.display = 'none';
    isThinking = false; 
});
document.getElementById('btn-start-load').addEventListener('click', () => {
    document.getElementById('import-file').click();
});

// 理쒖큹 ?ㅽ뻾 諛?珥덇린 ?곹깭
isThinking = true; // 紐⑤떖???リ린 ?꾧퉴吏??李⑹닔 ?쒗븳
initModel(); recalculateHeatmap(); renderBoard();
window.addEventListener('resize', () => {
    renderBoard();
});
