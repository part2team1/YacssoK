"""Local-only educational demo usage notice."""

NOTICE_HTML = """
<button type="button" class="notice-reopen">이용 안내</button>
<dialog class="usage-dialog" aria-labelledby="usage-title" aria-describedby="usage-summary">
  <div class="usage-content">
    <span class="usage-label">YACSSOK</span>
    <h2 id="usage-title">알약 인식 이용 안내</h2>
    <p id="usage-summary">사진 인식 결과와 식약처 공개 정보는 참고용으로 제공됩니다.</p>
    <ul>
      <li>AI가 알약을 잘못 인식할 수 있습니다.</li>
      <li>이 데모는 질병 진단이나 처방·복약지도를 대신하지 않습니다.</li>
      <li>약의 복용·중단·변경은 의사 또는 약사와 상담해주세요.</li>
    </ul>
    <button type="button" class="notice-confirm">이해했습니다</button>
  </div>
</dialog>
"""
NOTICE_CSS = """
.notice-reopen{display:block;margin:0 0 8px auto;background:none;border:0;color:#557268;font:inherit;font-size:13px;text-decoration:underline;cursor:pointer}
.usage-dialog{border:1px solid #dce5df;border-radius:24px;padding:0;width:min(500px,calc(100vw - 32px));max-height:calc(100dvh - 40px);box-sizing:border-box;color:#263b32;background:#fff;box-shadow:0 24px 90px #172f3333;overflow:auto}
.usage-dialog::backdrop{background:rgba(25,42,36,.52);backdrop-filter:blur(3px)}
.usage-content{padding:32px}.usage-label{font-size:12px;letter-spacing:2px;color:#567e6d}
.usage-content h2{font-size:25px;margin:10px 0 18px;line-height:1.35;letter-spacing:-.7px}
.usage-content p{font-size:15px;line-height:1.8;color:#56645c;margin:0 0 20px}
.usage-content ul{padding-left:20px;margin:0 0 26px}.usage-content li{font-size:15px;line-height:1.8;margin:12px 0}
.notice-confirm{width:100%;border:0;border-radius:12px;background:#337f73;color:white;min-height:50px;font:inherit;font-weight:700;cursor:pointer}.notice-confirm:focus-visible,.notice-reopen:focus-visible{outline:3px solid #83b7a8;outline-offset:3px}
@media(max-width:480px){.usage-content{padding:24px}.usage-content h2{font-size:22px}}
"""
NOTICE_JS = """
const dialog=element.querySelector('dialog');
const openButton=element.querySelector('.notice-reopen');
const confirmButton=element.querySelector('.notice-confirm');
const open=()=>{if(!dialog.open)dialog.showModal();confirmButton.focus();};
openButton.addEventListener('click',open);
confirmButton.addEventListener('click',()=>dialog.close());
dialog.addEventListener('close',()=>openButton.focus());
open();
"""
