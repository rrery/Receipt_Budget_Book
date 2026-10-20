const button=document.getElementById('print'),image=document.querySelector('img');
const ready=()=>{button.disabled=false;button.textContent='PDF로 저장 / 인쇄';};
if(!image.complete){button.disabled=true;button.textContent='달력을 준비하는 중…';image.addEventListener('load',ready);}
image.addEventListener('error',()=>{button.disabled=true;button.textContent='이미지를 불러오지 못했어요. 새로고침해주세요.';});
button.addEventListener('click',()=>window.print());
