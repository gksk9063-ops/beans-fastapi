// 분류 응답 JSON을 카드 형태로 보여준다. (API 응답은 바꾸지 않고 화면 표시만 담당)
const LABEL_KO = {angular_leaf_spot: '각진 잎 반점병', bean_rust: '콩 녹병', healthy: '정상'};
const REVIEW_SCORE = 0.9;  // 06 보고서 R1 규칙: 점수 0.9 미만이면 '확인 필요'

function pct(score) { return (score * 100).toFixed(1) + '%'; }

function renderResult(box, data, httpStatus) {
  box.hidden = false;
  if (!data || !data.top1) {          // 오류 응답 또는 아직 처리 중
    const msg = data && (data.detail || data.error_message || data.status) || '결과 없음';
    box.innerHTML = `<div class="card error">HTTP ${httpStatus ?? ''} · ${msg}</div>`;
    return;
  }
  const top = data.top1;
  const review = top.score < REVIEW_SCORE;
  const bars = data.all.map(r => `
    <div class="bar-row ${r.label === top.label ? 'top' : ''}">
      <span class="bar-name">${LABEL_KO[r.label] || r.label}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${(r.score * 100).toFixed(1)}%"></span></span>
      <span class="bar-val">${pct(r.score)}</span>
    </div>`).join('');
  box.innerHTML = `
    <div class="card ${review ? 'review' : ''}">
      <div class="card-head">
        <div>
          <div class="top-label">${LABEL_KO[top.label] || top.label}</div>
          <div class="top-sub">${top.label}</div>
        </div>
        <div class="top-score">${pct(top.score)}</div>
      </div>
      ${review ? `<div class="notice">확신도 ${pct(top.score)} &lt; 90% → 사람 확인 필요</div>` : ''}
      <div class="bars">${bars}</div>
      <div class="badges">
        <span class="badge ${data.cache_hit ? 'hit' : 'miss'}">캐시 ${data.cache_hit ? 'HIT' : 'MISS'}</span>
        <span class="badge">추론 ${data.inference_ms == null ? '생략(캐시)' : data.inference_ms.toFixed(1) + 'ms'}</span>
        <span class="badge">모델 ${String(data.model_id).split('/').pop()}</span>
        ${data.job_id ? `<span class="badge">job ${data.job_id} · ${data.status}</span>` : ''}
      </div>
    </div>`;
}
