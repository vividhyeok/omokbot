(function () {
    const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

    function formatCoord(idx, size) {
        if (!Number.isFinite(idx) || idx < 0) return '-';
        const x = idx % size;
        const y = Math.floor(idx / size);
        return `${x + 1}, ${y + 1}`;
    }

    function topArrayEntries(values, limit = 3) {
        return values
            .map((value, idx) => ({ idx, value: Number(value) || 0 }))
            .filter(item => item.value > 0)
            .sort((a, b) => b.value - a.value)
            .slice(0, limit);
    }

    function topObjectEntries(obj, limit = 3) {
        return Object.entries(obj || {})
            .map(([key, value]) => ({ key, value: Number(value) || 0 }))
            .filter(item => item.value > 0)
            .sort((a, b) => b.value - a.value)
            .slice(0, limit);
    }

    function renderTagList(elementId, items, fallback) {
        const el = document.getElementById(elementId);
        if (!el) return;
        if (!items.length) {
            el.innerHTML = `<span class="panel-note">${fallback}</span>`;
            return;
        }
        el.innerHTML = items.map(item => `<span class="memory-tag">${item}</span>`).join('');
    }

    function getMainDecisionReason(item) {
        if (item.selected && item.isWinningMove) return '마무리';
        if (item.isForcedBlock) return '방어 우선';
        if (item.riskAfter > 0) return '위험 관리';
        if (item.memory >= item.tactic && item.memory >= item.value && item.memory > 0.25) return '기억';
        if (item.value >= item.tactic && item.value > 0.35) return '경험';
        return '규칙';
    }

    function buildDecisionSummary(rankedMoves, selectedIdx, context = {}, size) {
        const top = rankedMoves.slice(0, 5);
        if (!top.length) return null;

        const maxTactic = Math.max(1, ...top.map(item => Math.abs(item.tacticContribution || item.rawScore || 0)));
        const maxMemory = Math.max(0.001, ...top.map(item => Math.max(0, item.predictionContribution || 0)));
        const maxValue = Math.max(0.001, ...top.map(item => Math.max(0, item.adaptiveContribution || 0)));
        const maxProb = Math.max(0.001, ...top.map(item => Number(item.prob) || 0));

        return {
            selectedIdx,
            candidates: top.map(item => {
                const idx = item.idx;
                const summary = {
                    idx,
                    coord: formatCoord(idx, size),
                    selected: idx === selectedIdx,
                    isWinningMove: Boolean(context.winningMove && idx === selectedIdx),
                    isForcedBlock: Boolean(context.forcedBlocks && context.forcedBlocks.has(idx)),
                    riskAfter: Number(item.riskAfter) || 0,
                    tactic: clamp(Math.abs(item.tacticContribution || item.rawScore || 0) / maxTactic, 0, 1),
                    memory: clamp(Math.max(0, item.predictionContribution || 0) / maxMemory, 0, 1),
                    value: clamp(Math.max(0, item.adaptiveContribution || 0) / maxValue, 0, 1),
                    policy: clamp((Number(item.prob) || 0) / maxProb, 0, 1),
                    probability: Number(item.prob) || 0
                };
                summary.reason = getMainDecisionReason(summary);
                return summary;
            })
        };
    }

    function renderDecisionPanel(summary) {
        const list = document.getElementById('decision-list');
        if (!list) return;
        if (!summary || !summary.candidates.length) {
            list.innerHTML = '<div class="panel-note">봇이 한 수를 둔 뒤 후보 분석이 여기에 표시됩니다.</div>';
            return;
        }

        list.innerHTML = summary.candidates.map(item => `
            <div class="decision-item ${item.selected ? 'is-selected' : ''}">
                <div class="decision-head">
                    <span>${item.selected ? '선택' : '후보'} ${item.coord}</span>
                    <span class="decision-reason">${item.reason} · ${Math.round(item.probability * 100)}%</span>
                </div>
                <div class="decision-bars">
                    <div class="decision-bar">
                        <span>규칙</span>
                        <span class="decision-bar-track"><span class="decision-bar-fill" style="width:${Math.round(item.tactic * 100)}%"></span></span>
                        <span>${Math.round(item.tactic * 100)}</span>
                    </div>
                    <div class="decision-bar">
                        <span>기억</span>
                        <span class="decision-bar-track"><span class="decision-bar-fill memory" style="width:${Math.round(item.memory * 100)}%"></span></span>
                        <span>${Math.round(item.memory * 100)}</span>
                    </div>
                    <div class="decision-bar">
                        <span>경험</span>
                        <span class="decision-bar-track"><span class="decision-bar-fill value" style="width:${Math.round(item.value * 100)}%"></span></span>
                        <span>${Math.round(item.value * 100)}</span>
                    </div>
                    <div class="decision-bar">
                        <span>최종</span>
                        <span class="decision-bar-track"><span class="decision-bar-fill policy" style="width:${Math.round(item.policy * 100)}%"></span></span>
                        <span>${Math.round(item.policy * 100)}</span>
                    </div>
                </div>
            </div>
        `).join('');
    }

    function updateLearningPanel(state) {
        const phaseEl = document.getElementById('learning-phase');
        if (!phaseEl) return;
        const activeStep = state.isReviewing
            ? 'review'
            : (!state.gameHistory.length && state.lastLearningSummary ? 'apply' : 'play');
        document.querySelectorAll('[data-learning-step]').forEach(step => {
            step.classList.toggle('is-active', step.dataset.learningStep === activeStep);
        });

        if (state.isReviewing) {
            phaseEl.innerText = '방금 판을 복습하며 다음 판에 반영할 기억을 고르고 있습니다.';
        } else if (!state.experimentSettings.learningEnabled) {
            phaseEl.innerText = '경험 저장을 잠시 멈춘 상태입니다.';
        } else if (state.isThinking) {
            phaseEl.innerText = '봇이 방금 장면을 보고 있습니다.';
        } else if (!state.gameHistory.length) {
            phaseEl.innerText = '새 판을 준비하고 있습니다.';
        } else {
            phaseEl.innerText = '이번 판의 흐름을 기억하며 두고 있습니다.';
        }

        document.getElementById('learning-confidence-bar').style.width = `${Math.round(state.confidence * 100)}%`;
        document.getElementById('learning-seen-scenes').innerText = `${state.gameHistory.length}개`;
        document.getElementById('learning-games').innerText = `${state.playerModel.gamesAnalyzed || 0}판`;

        const lastEl = document.getElementById('learning-last-summary');
        if (lastEl) {
            if (state.lastLearningSummary) {
                const lossText = Number.isFinite(state.lastLearningSummary.loss)
                    ? ` · 복습 오차 ${state.lastLearningSummary.loss.toFixed(4)}`
                    : '';
                lastEl.innerText = `${state.lastLearningSummary.resultLabel} 뒤 ${state.lastLearningSummary.learnedMoves}개 장면을 복습했습니다${lossText}.`;
            } else {
                lastEl.innerText = '아직 복습한 판이 없습니다.';
            }
        }

        const threatTags = topObjectEntries(state.playerModel.threatCells, 3)
            .map(item => `위험 ${formatCoord(Number(item.key), state.size)}`);
        const commonTags = topArrayEntries(state.playerModel.moveCount, 2)
            .map(item => `자주 둠 ${formatCoord(item.idx, state.size)}`);
        renderTagList('learning-tags', [...threatTags, ...commonTags].slice(0, 4), '기억할 패턴을 모으는 중입니다.');
    }

    function updateMemoryPanel(state) {
        renderTagList(
            'memory-common-moves',
            topArrayEntries(state.playerModel.moveCount, 5).map(item => `${formatCoord(item.idx, state.size)} · ${item.value}회`),
            '아직 충분한 위치 기억이 없습니다.'
        );
        renderTagList(
            'memory-threats',
            topObjectEntries(state.playerModel.threatCells, 5).map(item => `${formatCoord(Number(item.key), state.size)} · ${item.value}회`),
            '아직 특별히 조심하는 자리가 없습니다.'
        );

        const flowEl = document.getElementById('memory-flow');
        if (!flowEl) return;

        const flows = Object.entries(state.playerModel.transition || {})
            .flatMap(([from, next]) => Object.entries(next || {}).map(([to, value]) => ({
                from: Number(from),
                to: Number(to),
                value: Number(value) || 0
            })))
            .filter(item => item.value > 0)
            .sort((a, b) => b.value - a.value)
            .slice(0, 3);
        flowEl.innerText = flows.length
            ? flows.map(item => `${formatCoord(item.from, state.size)} -> ${formatCoord(item.to, state.size)} (${item.value}회)`).join(' / ')
            : '아직 반복 흐름이 충분하지 않습니다.';
    }

    function renderModalLearningSummary(summary) {
        const el = document.getElementById('modal-learning-summary');
        if (!el) return;
        if (!summary) {
            el.style.display = 'none';
            el.innerHTML = '';
            return;
        }
        el.style.display = 'block';
        const status = summary.learningEnabled ? '경험 저장 완료' : '경험 저장 꺼짐';
        const lossText = Number.isFinite(summary.loss) ? `<br>복습 오차: ${summary.loss.toFixed(4)}` : '';
        el.innerHTML = `
            <strong>${status}</strong><br>
            봇 착수 ${summary.learnedMoves}개, 사용자 착수 ${summary.userMoves}개를 확인했습니다.<br>
            다음 판에 더 조심할 자리: ${summary.threats.length ? summary.threats.join(', ') : '아직 없음'}${lossText}
        `;
    }

    function describeScale(value, middle = '보통') {
        const v = Number(value);
        if (v < 0.45) return '낮음';
        if (v < 0.85) return '약간 낮음';
        if (v <= 1.15) return middle;
        if (v <= 1.55) return '약간 높음';
        return '높음';
    }

    function syncExperimentControls(settings) {
        [
            ['setting-adventure', 'adventure'],
            ['setting-tactics', 'tactics'],
            ['setting-memory', 'memory'],
            ['setting-learning-speed', 'learningSpeed']
        ].forEach(([id, key]) => {
            const input = document.getElementById(id);
            if (input) input.value = settings[key];
        });

        const enabled = document.getElementById('setting-learning-enabled');
        if (enabled) enabled.checked = Boolean(settings.learningEnabled);
        const preset = document.getElementById('experiment-preset');
        if (preset) preset.value = settings.preset || 'balanced';

        const labelMap = {
            'value-adventure': describeScale(settings.adventure),
            'value-tactics': describeScale(settings.tactics),
            'value-memory': describeScale(settings.memory),
            'value-learning-speed': describeScale(settings.learningSpeed)
        };
        Object.entries(labelMap).forEach(([id, text]) => {
            const el = document.getElementById(id);
            if (el) el.innerText = text;
        });
    }

    function initExperimentControls({ settings, onChange }) {
        const presets = {
            balanced: { adventure: 1, tactics: 1, memory: 1, learningSpeed: 1 },
            beginner: { adventure: 1.7, tactics: 0.7, memory: 0.8, learningSpeed: 1.2 },
            tactical: { adventure: 0.45, tactics: 1.8, memory: 0.5, learningSpeed: 0.9 },
            memory: { adventure: 0.8, tactics: 0.8, memory: 1.8, learningSpeed: 1.15 }
        };

        syncExperimentControls(settings);

        const notifyChange = () => {
            syncExperimentControls(settings);
            if (typeof onChange === 'function') onChange(settings);
        };

        const preset = document.getElementById('experiment-preset');
        if (preset) {
            preset.addEventListener('change', () => {
                Object.assign(settings, presets[preset.value] || presets.balanced, { preset: preset.value });
                notifyChange();
            });
        }

        [
            ['setting-adventure', 'adventure'],
            ['setting-tactics', 'tactics'],
            ['setting-memory', 'memory'],
            ['setting-learning-speed', 'learningSpeed']
        ].forEach(([id, key]) => {
            const input = document.getElementById(id);
            if (!input) return;
            input.addEventListener('input', () => {
                settings[key] = Number(input.value);
                settings.preset = 'custom';
                notifyChange();
            });
        });

        const enabled = document.getElementById('setting-learning-enabled');
        if (enabled) {
            enabled.addEventListener('change', () => {
                settings.learningEnabled = enabled.checked;
                notifyChange();
            });
        }
    }

    window.OmokEducation = {
        buildDecisionSummary,
        formatCoord,
        initExperimentControls,
        renderDecisionPanel,
        renderModalLearningSummary,
        syncExperimentControls,
        updateLearningPanel,
        updateMemoryPanel
    };
})();
