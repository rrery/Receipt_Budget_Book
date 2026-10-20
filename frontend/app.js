const $ = id => document.getElementById(id);
const money = value => Number(value || 0).toLocaleString('ko-KR');
const esc = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const dateKey = value => `${value.getFullYear()}-${String(value.getMonth()+1).padStart(2,'0')}-${String(value.getDate()).padStart(2,'0')}`;
const today = dateKey(new Date());
let month=today.slice(0,7), selected=null, filter='all', limit=60, editing=null;
let session={authenticated:false,mode:'local'}, categoryNames=[], summary=null;
let pollTimer,toastTimer,requestVersion=0,memoDirty=false,signup=false,shareURL=null,shareFile=null,shareVersion=0;
const uploads=new Map();
const safe=fn=>async(...args)=>{try{await fn(...args);}catch(error){toast(error.message);}};

async function api(path,options={}){
  const response=await fetch('/api'+path,{credentials:'same-origin',...options});
  if(!response.ok){
    let body={};try{body=await response.json();}catch{}
    if(response.status===401&&!path.startsWith('/auth/'))showAuth();
    throw new Error(typeof body.detail==='string'?body.detail:'입력을 확인해주세요. 메모는 200자까지 쓸 수 있어요.');
  }
  return response.status===204?null:response.json();
}
function toast(message){$('toast').textContent=message;$('toast').classList.remove('hidden');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.add('hidden'),5000);}
function ensureAuth(){if(session.authenticated)return true;showAuth();return false;}
function emptySummary(){return {month,total:0,count:0,days:[],reviewed_count:0,unknown_amount_count:0,pending_count:0,categories:[],previous_month:{total:0,count:0,difference:0,percent:null},memo:''};}
async function refresh(){
  const version=++requestVersion,requestedMonth=month;
  let ledger=emptySummary(),records={items:[],total:0},status={pending_count:0};
  if(session.authenticated){
    let query=`?month=${month}&limit=${limit}`;
    if(selected)query+='&purchased_at='+selected;
    if(filter==='review')query+='&status=review';
    [ledger,records,status]=await Promise.all([api('/calendar?month='+month),api('/receipts'+query),api('/status')]);
  }
  if(version!==requestVersion||month!==requestedMonth)return;
  summary=ledger;renderCalendar(ledger);renderInsights(ledger);renderRecords(records);
  if(!memoDirty)$('month-memo').value=ledger.memo||'';
  updateMemoCount();
  $('all').classList.toggle('active',filter==='all');$('review-filter').classList.toggle('active',filter==='review');
  $('review-count').textContent=ledger.count-ledger.reviewed_count;
  $('clear-date').classList.toggle('hidden',!selected);
  $('list-title').textContent=selected?`${Number(selected.slice(-2))}일의 소비 기록`:'이번 달의 영수증';
  if(status.pending_count||ledger.pending_count||uploads.size)schedulePoll();
}
function shortAmount(value){const unit=value>=1e12?[1e12,'조']:value>=1e8?[1e8,'억']:value>=1e5?[1e4,'만']:null;return unit?`${(value/unit[0]).toLocaleString('ko-KR',{maximumFractionDigits:1})}${unit[1]}`:money(value);}
function renderCalendar(ledger){
  const [year,m]=month.split('-').map(Number),first=new Date(year,m-1,1).getDay(),last=new Date(year,m,0).getDate();
  $('month-label').textContent=`${year}년 ${m}월`;
  $('month-mood').textContent=ledger.count?`소중한 기록 ${ledger.count}장`:'차곡차곡 쌓이는 중';
  const days=new Map(ledger.days.map(day=>[day.date,day]));
  $('calendar').innerHTML=Array.from({length:Math.ceil((first+last)/7)*7},(_,index)=>{
    const day=index-first+1,weekend=index%7===0?'sun':index%7===6?'sat':'';
    if(day<1||day>last)return `<div class="day blank ${weekend}" aria-hidden="true"></div>`;
    const key=`${month}-${String(day).padStart(2,'0')}`,entry=days.get(key);
    const description=entry?`${money(entry.total)}원, 영수증 ${entry.count}장${entry.pending?', 분석 중':''}${entry.unknown_amount?', 금액 미확인 포함':''}`:'기록 없음';
    return `<button class="day ${weekend} ${key===today?'today':''} ${key===selected?'selected':''}" data-date="${key}" aria-label="${key}, ${description}" aria-pressed="${key===selected}"><span class="number">${day}</span>${entry?.unreviewed?'<span class="review-dot" title="아직 안 살펴본 기록"></span>':''}${entry?`${entry.total||!entry.unknown_amount?`<span class="day-total" title="${money(entry.total)}원">${shortAmount(entry.total)}<span class="day-won">원</span></span>`:''}<span class="day-count">${entry.count}장</span>${entry.pending?'<span class="day-pending">읽고 있어요…</span>':entry.unknown_amount?'<span class="day-pending">금액 미확인</span>':''}`:''}</button>`;
  }).join('');
  $('calendar').querySelectorAll('button').forEach(button=>button.onclick=safe(async()=>{
    selected=button.dataset.date;limit=60;await refresh();$('receipt-list').scrollIntoView({behavior:'smooth',block:'center'});
  }));
}
function renderInsights(ledger){
  $('month-total').innerHTML=money(ledger.total)+'<small>원</small>';
  $('month-count').innerHTML=ledger.count+'<small>장</small>';
  $('reviewed-count').innerHTML=ledger.reviewed_count+'<small>장</small>';
  $('review-caption').textContent=ledger.count?`${ledger.count-ledger.reviewed_count}장은 필요할 때 살펴봐요`:'꼭 검토하지 않아도 기록돼요';
  $('top-category').textContent=ledger.top_category?.category||'기록을 모으는 중이에요';
  $('top-category-amount').textContent=ledger.top_category?`${money(ledger.top_category.total)}원 · 나만의 소비 취향`:'분류별로 나의 취향을 발견해요.';
  $('top-day').textContent=ledger.top_day?`${Number(ledger.top_day.date.slice(5,7))}월 ${Number(ledger.top_day.date.slice(-2))}일`:'아직은 빈 페이지';
  $('top-day-amount').textContent=ledger.top_day?`${money(ledger.top_day.total)}원 · 조금 특별했던 하루`:'어떤 하루가 기록될까요?';
  const prev=ledger.previous_month,more=prev.difference>0;
  let comparison='비교할 기록이 없어요',detail='이번 달부터 차근차근 시작해요.';
  if(prev.count){
    comparison=prev.percent!==null?`${Math.abs(prev.percent).toFixed(1)}% ${more?'더':'덜'} 썼어요`:`${money(Math.abs(prev.difference))}원 ${more?'더':'덜'} 썼어요`;
    if(prev.difference===0)comparison='지난달과 같아요';
    detail=`지난달 ${money(prev.total)}원 · ${money(Math.abs(prev.difference))}원 차이`;
  }
  $('comparison').textContent=comparison;$('comparison-detail').textContent=detail;
  $('comparison-caption').textContent=prev.count?`지난달보다 ${money(Math.abs(prev.difference))}원 ${more?'더':'덜'} 썼어요`:'하나씩 채워가는 나의 소비 기록';
  $('unknown-notice').classList.toggle('hidden',!ledger.unknown_amount_count&&!ledger.demo_count);
  $('unknown-notice').textContent=(ledger.unknown_amount_count?`금액 미확인 ${ledger.unknown_amount_count}장은 달력에 표시되며, 금액을 알게 되면 합계에 더해져요. `:'')+(ledger.demo_count?`샘플 ${ledger.demo_count}장이 통계에 포함되어 있어요.`:'');
  let categories=ledger.categories.filter(c=>c.total>0);
  if(categories.length>5)categories=[...categories.slice(0,4),{category:'그 외',total:categories.slice(4).reduce((sum,c)=>sum+c.total,0)}];
  $('category-chart').innerHTML=categories.length?categories.map(entry=>`<div class="category-row"><span class="category-label">${esc(entry.category)}</span><div class="category-track" role="img" aria-label="${esc(entry.category)} ${money(entry.total)}원"><div class="category-fill" style="width:${Math.max(3,entry.total/ledger.total*100)}%"></div></div><span class="category-amount">${money(entry.total)}원</span></div>`).join(''):'<div class="empty">나의 소비 취향을 차곡차곡 알아가요.</div>';
}
const categoryEmoji={'식비':'🍚','카페·간식':'☕','장보기':'🥕','교통':'🚌','쇼핑':'🛍','생활':'🧺','의료·건강':'🌿','문화·여가':'🎬','기타':'🧾'};
function renderRecords(records){
  $('receipt-list').innerHTML=records.items.length?records.items.map(row=>{
    const busy=['queued','processing'].includes(row.job_status),failed=row.job_status==='failed';
    const badge=busy?(row.job_status==='queued'?'분석 대기':'읽는 중…'):failed?'읽기 실패':row.status==='confirmed'?'살펴봤어요':'자동 기록';
    const source={demo:'샘플',manual:'직접 기록',upload:'영수증'}[row.source]||'';
    const iconClass=row.category==='식비'?'food':row.category==='카페·간식'?'cafe':row.category==='쇼핑'?'shopping':'';
    return `<article class="receipt-row"><span class="receipt-icon ${iconClass}" aria-hidden="true">${categoryEmoji[row.category]||'🧾'}</span><div class="receipt-info"><b>${esc(row.store_name||'새 영수증을 읽고 있어요')}</b><small>${esc(row.purchased_at||'날짜 미확인')}${row.date_inferred?' · 임시 날짜':''} · ${esc(row.category)} · ${source}</small>${row.memo?`<small class="receipt-memo">✎ ${esc(row.memo)}</small>`:''}${failed?'<small class="receipt-error">다시 읽거나 직접 수정할 수 있어요.</small>':''}</div><span class="badge ${failed?'failed':row.status!=='confirmed'?'unreviewed':''}">${badge}</span><span class="receipt-money">${row.total_amount===null?'금액 미확인':money(row.total_amount)+'원'}</span>${!busy?`${failed?`<button class="receipt-action retry" data-retry="${row.id}">다시 읽기</button>`:''}<button class="receipt-action" data-edit="${row.id}">${row.status==='confirmed'?'열어보기':'살펴보기'} ↗</button>`:''}</article>`;
  }).join(''):`<div class="empty"><span class="empty-doodle">▤ ˘ ▤</span>${selected?'이날은 아직 기록이 없어요.':'아직은 깨끗한 나의 달력.'}<br>영수증 한 장이나 작은 소비로 채워볼까요?</div>`;
  $('receipt-list').querySelectorAll('[data-edit]').forEach(button=>button.onclick=safe(()=>openEditor(button.dataset.edit)));
  $('receipt-list').querySelectorAll('[data-retry]').forEach(button=>button.onclick=safe(async()=>{await api('/receipts/'+button.dataset.retry+'/retry',{method:'POST'});toast('영수증을 다시 읽을게요.');await refresh();}));
  $('more').classList.toggle('hidden',records.total<=records.items.length||limit>=500);
}
function navigateMonth(delta){const [year,m]=month.split('-').map(Number);const value=new Date(year,m-1+delta,1);month=dateKey(value).slice(0,7);selected=null;limit=60;memoDirty=false;return refresh();}
$('prev').onclick=safe(()=>navigateMonth(-1));$('next').onclick=safe(()=>navigateMonth(1));
$('today').onclick=safe(async()=>{month=today.slice(0,7);selected=null;limit=60;memoDirty=false;await refresh();});
$('all').onclick=safe(async()=>{filter='all';selected=null;limit=60;await refresh();});
$('review-filter').onclick=safe(async()=>{filter='review';limit=60;await refresh();});
$('clear-date').onclick=safe(async()=>{selected=null;await refresh();});
$('more').onclick=safe(async()=>{limit=Math.min(limit+60,500);await refresh();});
$('demo').onclick=safe(async()=>{if(!ensureAuth())return;const result=await api('/demo',{method:'POST'});month=result.month;selected=null;filter='all';memoDirty=false;toast(result.message);await refresh();});
function updateMemoCount(){$('month-memo-count').textContent=`${[...$('month-memo').value].length} / 200`;}
$('month-memo').oninput=()=>{$('month-memo').value=[...$('month-memo').value].slice(0,200).join('');memoDirty=true;updateMemoCount();};
$('save-month-memo').onclick=safe(async()=>{if(!ensureAuth())return;$('save-month-memo').disabled=true;try{await api('/months/'+month+'/memo',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({memo:$('month-memo').value})});memoDirty=false;toast('이번 달의 한마디를 간직했어요.');}finally{$('save-month-memo').disabled=false;}});

$('upload').onclick=()=>{if(ensureAuth())$('file').click();};
$('file').onchange=safe(event=>uploadFiles([...event.target.files]));
['dragenter','dragover'].forEach(name=>$('drop').addEventListener(name,event=>{event.preventDefault();$('drop').classList.add('drag');}));
['dragleave','drop'].forEach(name=>$('drop').addEventListener(name,event=>{event.preventDefault();$('drop').classList.remove('drag');}));
$('drop').addEventListener('drop',safe(event=>uploadFiles([...event.dataTransfer.files])));
async function uploadFiles(files){
  if(!files.length||!ensureAuth())return;
  if(files.length>20){toast('한 번에 사진 20장까지 올릴 수 있어요.');return;}
  $('upload').disabled=true;$('account').disabled=true;$('upload-status').className='upload-banner';
  let count=0,failures=0;
  for(const file of files){
    $('upload-status').textContent=`사진을 달력에 담고 있어요… (${count+failures+1}/${files.length})`;
    try{
      if(file.size>12*1024*1024)throw new Error('사진은 12MB 이하로 올려주세요.');
      const form=new FormData();form.append('file',file);
      const result=await api('/receipts/upload',{method:'POST',body:form});
      if(!result.duplicate)uploads.set(result.job_id,file.name);
      count++;
    }catch(error){failures++;toast(file.name+': '+error.message);}
  }
  $('file').value='';$('upload').disabled=false;$('account').disabled=false;
  $('upload-status').textContent=`사진 ${count}장을 담았어요.${failures?' '+failures+'장은 업로드하지 못했어요.':''} 읽는 동안 오늘 날짜에 임시로 표시되고, 구매 날짜를 찾으면 옮겨져요.`;
  month=today.slice(0,7);selected=null;filter='all';memoDirty=false;
  await refresh();schedulePoll();
}
function schedulePoll(){clearTimeout(pollTimer);pollTimer=setTimeout(safe(async()=>{await pollUploads();await refresh();}),3000);}
async function pollUploads(){
  for(const [jobId] of uploads){
    const job=await api('/jobs/'+jobId);
    if(job.status==='completed'){
      uploads.delete(jobId);const row=await api('/receipts/'+job.receipt_id);
      $('upload-status').replaceChildren(document.createTextNode('사진을 읽고 달력에 기록했어요. '));
      const button=document.createElement('button');button.className='text-button';button.textContent=(row.purchased_at||'').replaceAll('-','.')+' 기록 보기 ↗';
      button.onclick=safe(async()=>{if(row.purchased_at){month=row.purchased_at.slice(0,7);selected=row.purchased_at;memoDirty=false;}await refresh();await openEditor(row.id);});
      $('upload-status').append(button);
    }else if(job.status==='failed'){uploads.delete(jobId);toast('사진을 다 읽지 못했어요. 달력에서 다시 읽거나 직접 수정할 수 있어요.');}
  }
}
function addItem(value={}){
  const row=document.createElement('div');row.className='item-row';
  row.innerHTML=`<input class="name" aria-label="품목명" maxlength="200" required placeholder="품목명" value="${esc(value.name||'')}"><input class="qty" type="number" min="1" max="100000" step="1" aria-label="수량" value="${esc(value.quantity??'')}"><input class="unit" type="number" min="0" step="1" aria-label="단가" value="${esc(value.unit_price??'')}"><input class="line" type="number" min="0" step="1" aria-label="품목 금액" value="${esc(value.line_total??'')}"><button type="button" aria-label="품목 삭제">×</button>`;
  row.querySelector('button').onclick=()=>row.remove();$('items').append(row);
}
$('add-item').onclick=()=>addItem();
async function openEditor(id=null){
  if(!ensureAuth())return;
  const row=id?await api('/receipts/'+id):null;editing=id;
  $('receipt-form').reset();$('items').replaceChildren();$('edit-error').textContent='';
  $('edit-title').textContent=row?'나의 소비 기록':'새로운 작은 소비';
  $('store-name').value=row?.store_name||'';$('purchase-date').value=row?(row.purchased_at||''):(selected||today);
  $('amount').value=row?.total_amount??'';$('category').value=row?.category||'기타';$('receipt-memo').value=row?.memo||'';
  $('receipt-memo-count').textContent=`${[...$('receipt-memo').value].length} / 200`;
  $('reviewed').checked=true;$('allow-training').checked=Boolean(row?.allow_training);
  $('training-consent-label').classList.toggle('hidden',!row?.has_image);
  $('source-label').textContent=row?{demo:'샘플 기록 · 실제 구매 내역이 아니에요',upload:'달력에 자동으로 기록했어요',manual:'내가 직접 남긴 기록'}[row.source]:'나를 위한 소비 기록';
  $('analysis-warnings').textContent=(row?.raw?.warnings||[]).join(' ');
  if(row?.date_inferred&&!$('analysis-warnings').textContent)$('analysis-warnings').textContent='구매 날짜를 읽지 못해 업로드 날짜에 임시로 기록했어요.';
  $('image-panel').classList.toggle('hidden',!row?.has_image);
  if(row?.has_image)$('receipt-image').src='/api/receipts/'+id+'/image';else $('receipt-image').removeAttribute('src');
  $('delete').classList.toggle('hidden',!row);for(const item of row?.items||[])addItem(item);
  $('history-details').classList.add('hidden');$('history-details').open=false;$('history').replaceChildren();
  if(row){const history=await api('/receipts/'+id+'/corrections');renderHistory(history);}
  if(!$('editor').open)$('editor').showModal();
}
function renderHistory(history){
  const labels={store_name:'상호명',purchased_at:'구매 날짜',total_amount:'총액',category:'분류',items:'품목'};
  $('history-details').classList.toggle('hidden',!history.length);
  $('history').innerHTML=history.map(entry=>`<div class="history-entry"><b>${esc(new Date(entry.created_at).toLocaleString('ko-KR'))}</b>${entry.changed_fields.map(key=>`<div>${labels[key]||key}: ${key==='items'?esc(entry.before.items.map(i=>i.name).join(', ')+' → '+entry.after.items.map(i=>i.name).join(', ')):esc((entry.before[key]??'미확인')+' → '+(entry.after[key]??'미확인'))}</div>`).join('')}</div>`).join('');
}
$('manual').onclick=safe(()=>openEditor());$('close-editor').onclick=()=>$('editor').close();
$('receipt-memo').oninput=()=>{$('receipt-memo').value=[...$('receipt-memo').value].slice(0,200).join('');$('receipt-memo-count').textContent=`${[...$('receipt-memo').value].length} / 200`;};
$('receipt-form').onsubmit=async event=>{
  event.preventDefault();$('save').disabled=true;$('edit-error').textContent='';
  const optional=value=>value===''?null:Number(value);
  const body={store_name:$('store-name').value,purchased_at:$('purchase-date').value||null,total_amount:optional($('amount').value),items:[...$('items').children].map(row=>({name:row.querySelector('.name').value,quantity:optional(row.querySelector('.qty').value),unit_price:optional(row.querySelector('.unit').value),line_total:optional(row.querySelector('.line').value)})),status:$('reviewed').checked?'confirmed':'review',memo:$('receipt-memo').value,category:$('category').value,allow_training:$('allow-training').checked};
  try{await api(editing?'/receipts/'+editing:'/receipts',{method:editing?'PATCH':'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});$('editor').close();if(body.purchased_at&&month!==body.purchased_at.slice(0,7)){month=body.purchased_at.slice(0,7);selected=null;memoDirty=false;}toast('기록을 차곡 담았어요.');await refresh();}catch(error){$('edit-error').textContent=error.message;}finally{$('save').disabled=false;}
};
$('delete').onclick=safe(async()=>{if(!editing||!confirm('이 영수증과 메모, 수정 기록을 삭제할까요?'))return;await api('/receipts/'+editing,{method:'DELETE'});$('editor').close();toast('기록을 지웠어요.');await refresh();});

function sharePath(extension='jpg'){return `/months/${month}/export.${extension}?include_memo=${$('include-memo').checked}`;}
async function updateSharePreview(){
  const version=++shareVersion;shareFile=null;
  $('share-error').textContent='';$('download-jpeg').disabled=true;$('native-share').disabled=true;
  try{const response=await fetch('/api'+sharePath('png'));if(!response.ok)throw new Error('이미지를 준비하지 못했어요. 잠시 후 다시 시도해주세요.');const blob=await response.blob();if(version!==shareVersion)return;if(shareURL)URL.revokeObjectURL(shareURL);shareURL=URL.createObjectURL(blob);$('share-preview').src=shareURL;shareFile=new File([blob],`차곡-${month}.png`,{type:'image/png'});$('native-share').classList.toggle('hidden',!navigator.canShare?.({files:[shareFile]}));}finally{if(version===shareVersion){$('download-jpeg').disabled=false;$('native-share').disabled=false;}}
}
$('share').onclick=safe(async()=>{if(!ensureAuth())return;$('share-dialog').showModal();try{await updateSharePreview();}catch(error){$('share-error').textContent=error.message;}});
$('close-share').onclick=()=>{$('share-dialog').close();};
$('include-memo').onchange=safe(updateSharePreview);
async function getJPEG(){const response=await fetch('/api'+sharePath());if(!response.ok)throw new Error('사진을 저장하지 못했어요.');return response.blob();}
$('download-jpeg').onclick=safe(async()=>{const blob=await getJPEG(),url=URL.createObjectURL(blob),anchor=document.createElement('a');anchor.href=url;anchor.download=`차곡-${month}.jpg`;document.body.append(anchor);anchor.click();anchor.remove();setTimeout(()=>URL.revokeObjectURL(url),2000);toast('이번 달을 한 장에 담았어요.');});
$('print-pdf').onclick=()=>{window.open(`/share/${month}?include_memo=${$('include-memo').checked}`,'_blank','noopener,noreferrer');};
if(navigator.share&&navigator.canShare)$('native-share').classList.remove('hidden');
$('native-share').onclick=safe(async()=>{if(!shareFile||!navigator.canShare({files:[shareFile]})){toast('이 기기에서는 JPEG 저장 버튼을 이용해주세요.');return;}try{await navigator.share({files:[shareFile],title:'나의 한 달, 차곡'});}catch(error){if(error.name!=='AbortError')throw error;}});

function showAuth(){signup=false;updateAuthLabels();$('auth-error').textContent='';if(!$('auth-dialog').open)$('auth-dialog').showModal();}
function updateAuthLabels(){$('auth-title').textContent=signup?'나만의 기록 시작하기':'나만의 가계부에 들어가기';$('auth-submit').textContent=signup?'가입하고 시작하기':'로그인';$('auth-toggle').textContent=signup?'이미 기록하고 있다면, 로그인 ↗':'처음이라면, 회원가입하기 ↗';$('password').autocomplete=signup?'new-password':'current-password';}
$('close-auth').onclick=()=>$('auth-dialog').close();$('auth-toggle').onclick=()=>{signup=!signup;$('auth-error').textContent='';updateAuthLabels();};
$('account').onclick=safe(async()=>{if(!session.authenticated){showAuth();return;}if(session.mode==='public'){await api('/auth/logout',{method:'POST'});session=await api('/auth/me');memoDirty=false;uploads.clear();shareFile=null;clearTimeout(pollTimer);if(shareURL){URL.revokeObjectURL(shareURL);shareURL=null;}$('share-preview').removeAttribute('src');updateAccount();await refresh();toast('편안한 하루 보내세요.');}});
function updateAccount(){$('account-name').textContent=session.authenticated?(session.mode==='public'?session.username+' · 로그아웃':'나의 기록'):'로그인';}
$('auth-form').onsubmit=async event=>{
  event.preventDefault();$('auth-submit').disabled=true;$('auth-error').textContent='';
  try{await api('/auth/'+(signup?'register':'login'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:$('username').value,password:$('password').value})});$('password').value='';session=await api('/auth/me');updateAccount();$('auth-dialog').close();memoDirty=false;await refresh();toast('반가워요. 오늘의 기록을 시작해볼까요?');}catch(error){$('auth-error').textContent=error.message;}finally{$('auth-submit').disabled=false;}
};
safe(async()=>{[session,categoryNames]=await Promise.all([api('/auth/me'),api('/categories')]);$('category').innerHTML=categoryNames.map(name=>`<option value="${esc(name)}">${esc(name)}</option>`).join('');updateAccount();await refresh();})();
setInterval(safe(async()=>{if(!$('editor').open&&!$('auth-dialog').open)await refresh();}),30000);
