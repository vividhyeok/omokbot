(function () {
    const COLUMN_LABELS = 'ABCDEFGHIJKLMNO'.split('');
    const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
    const escapeHtml = value => String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');

    function formatCoord(idx, size) {
        if (!Number.isFinite(idx) || idx < 0) return '-';
        const x = idx % size;
        const y = Math.floor(idx / size);
        return `${y + 1}. ${COLUMN_LABELS[x] || x + 1}`;
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
        el.innerHTML = items.map(item => {
            const text = escapeHtml(item);
            return `<span class="memory-tag" data-rl-tip="강화학습에서 이 태그는 모델이 기억한 상태 특징입니다. ${text} 같은 위치나 흐름이 다음 판단에 참고됩니다.">${text}</span>`;
        }).join('');
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
            <div class="decision-item ${item.selected ? 'is-selected' : ''}" data-rl-tip="강화학습에서 ${escapeHtml(item.coord)} 후보는 행동입니다. 정책은 여러 행동 후보 중 하나를 고릅니다.">
                <div class="decision-head">
                    <span>${item.selected ? '선택' : '후보'} ${item.coord}</span>
                    <span class="decision-reason">${item.reason} · ${Math.round(item.probability * 100)}%</span>
                </div>
                <div class="decision-bars">
                    <div class="decision-bar" data-rl-tip="강화학습에서 규칙 점수는 보상 설계입니다. 좋은 모양과 위험한 모양을 사람이 정한 기준으로 점수화합니다.">
                        <span>규칙</span>
                        <span class="decision-bar-track"><span class="decision-bar-fill" style="width:${Math.round(item.tactic * 100)}%"></span></span>
                        <span>${Math.round(item.tactic * 100)}</span>
                    </div>
                    <div class="decision-bar" data-rl-tip="강화학습에서 기억 점수는 경험 메모리입니다. 이전 판에서 자주 본 사용자 패턴을 참고합니다.">
                        <span>기억</span>
                        <span class="decision-bar-track"><span class="decision-bar-fill memory" style="width:${Math.round(item.memory * 100)}%"></span></span>
                        <span>${Math.round(item.memory * 100)}</span>
                    </div>
                    <div class="decision-bar" data-rl-tip="강화학습에서 경험 점수는 가치 함수입니다. 지난 복습을 바탕으로 이 상태가 좋아 보이는지 추정합니다.">
                        <span>경험</span>
                        <span class="decision-bar-track"><span class="decision-bar-fill value" style="width:${Math.round(item.value * 100)}%"></span></span>
                        <span>${Math.round(item.value * 100)}</span>
                    </div>
                    <div class="decision-bar" data-rl-tip="강화학습에서 최종 점수는 정책 분포입니다. 여러 신호를 합쳐 실제 선택 확률을 만듭니다.">
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

    function describeScale(value, labels = ['낮음', '약간 낮음', '보통', '약간 높음', '높음']) {
        const v = Number(value);
        if (v < 0.45) return labels[0];
        if (v < 0.85) return labels[1];
        if (v <= 1.15) return labels[2];
        if (v <= 1.55) return labels[3];
        return labels[4];
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
            'value-adventure': describeScale(settings.adventure, ['거의 안 둠', '가끔', '보통', '자주', '매우 자주']),
            'value-tactics': describeScale(settings.tactics, ['약하게', '조금 약하게', '보통', '강하게', '매우 강하게']),
            'value-memory': describeScale(settings.memory, ['약하게', '조금 약하게', '보통', '강하게', '매우 강하게']),
            'value-learning-speed': describeScale(settings.learningSpeed, ['천천히', '조금 천천히', '보통', '빠르게', '매우 빠르게'])
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

    function initRlTooltips() {
        const tooltip = document.getElementById('rl-tooltip');
        if (!tooltip || tooltip.dataset.bound === 'true') return;
        tooltip.dataset.bound = 'true';
        let currentTarget = null;

        const placeTooltip = (x, y) => {
            const margin = 12;
            tooltip.style.left = '0px';
            tooltip.style.top = '0px';
            const rect = tooltip.getBoundingClientRect();
            const nextX = clamp(x + 14, margin, window.innerWidth - rect.width - margin);
            const nextY = clamp(y + 14, margin, window.innerHeight - rect.height - margin);
            tooltip.style.left = `${nextX}px`;
            tooltip.style.top = `${nextY}px`;
        };

        const showTooltip = (target, event) => {
            const text = target.getAttribute('data-rl-tip');
            if (!text) return;
            currentTarget = target;
            tooltip.textContent = text;
            tooltip.classList.add('is-visible');
            target.setAttribute('aria-describedby', 'rl-tooltip');

            const rect = target.getBoundingClientRect();
            const x = event && Number.isFinite(event.clientX) ? event.clientX : rect.left + rect.width / 2;
            const y = event && Number.isFinite(event.clientY) ? event.clientY : rect.top + rect.height / 2;
            placeTooltip(x, y);
        };

        const hideTooltip = () => {
            if (currentTarget) currentTarget.removeAttribute('aria-describedby');
            currentTarget = null;
            tooltip.classList.remove('is-visible');
        };

        document.addEventListener('mouseover', event => {
            const target = event.target.closest('[data-rl-tip]');
            if (!target || target === currentTarget) return;
            showTooltip(target, event);
        });

        document.addEventListener('mousemove', event => {
            if (!currentTarget || !currentTarget.matches(':hover')) return;
            placeTooltip(event.clientX, event.clientY);
        });

        document.addEventListener('mouseout', event => {
            if (!currentTarget) return;
            const nextTarget = event.relatedTarget;
            if (nextTarget && currentTarget.contains(nextTarget)) return;
            hideTooltip();
        });

        document.addEventListener('focusin', event => {
            const target = event.target.closest('[data-rl-tip]');
            if (target) showTooltip(target, null);
        });

        document.addEventListener('focusout', event => {
            if (currentTarget && currentTarget.contains(event.target)) hideTooltip();
        });

        document.addEventListener('keydown', event => {
            if (event.key === 'Escape') hideTooltip();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initRlTooltips, { once: true });
    } else {
        initRlTooltips();
    }

    window.OmokEducation = {
        buildDecisionSummary,
        formatCoord,
        initExperimentControls,
        initRlTooltips,
        renderDecisionPanel,
        renderModalLearningSummary,
        syncExperimentControls,
        updateLearningPanel,
        updateMemoryPanel
    };
})();
