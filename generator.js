const gstate={token:"",plan:null,output:null,poller:null,resumeSessionId:null};
const $=id=>document.getElementById(id);
const escapeHtml=value=>String(value??"").replace(/[&<>'"]/g,char=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
function toast(message){const node=$("toast");node.textContent=message;node.classList.add("show");clearTimeout(toast.timer);toast.timer=setTimeout(()=>node.classList.remove("show"),2600);}
function step(name){document.querySelectorAll(".page-step").forEach(node=>node.classList.toggle("active",node.id===`gstep-${name}`));document.querySelectorAll("[data-gstep]").forEach(node=>node.classList.toggle("active",node.dataset.gstep===name));window.scrollTo({top:0,behavior:"smooth"});}
async function post(path,body){const response=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json","X-GPT56-Session":gstate.token},body:JSON.stringify(body)});const value=await response.json();if(!response.ok)throw new Error(value.error||`HTTP ${response.status}`);return value;}

function buildPlan(){
  const temporal=$("g-temporal").checked;
  const requestFormats=[$("g-format-normal").checked&&"normal",$("g-format-native").checked&&"native_codex"].filter(Boolean);
  const contextModes=[$("g-context-none").checked&&"no_history",$("g-context-32k").checked&&"fixed_32k_history"].filter(Boolean);
  if(!requestFormats.length||!contextModes.length)throw new Error("至少选择一种请求外观和一种上下文");
  return {
    name:$("g-name").value.trim(),probe_id:$("g-probe-id").value.trim(),user_prompt:$("g-prompt").value,
    model_names:{"gpt-5.6-sol":$("g-sol").value.trim(),"gpt-5.6-terra":$("g-terra").value.trim(),"gpt-5.6-luna":$("g-luna").value.trim()},
    effort:$("g-effort").value,samples_per_model:Number($("g-samples").value),runtime_samples:Number($("g-runtime-samples").value),
    developer_prompt:$("g-developer").value,request_formats:requestFormats,context_modes:contextModes,
    normalizer:{id:$("g-normalizer").value,parameters:{}},temporal_windows:temporal?4:1,
    window_gap_seconds:Number($("g-gap").value),workers:Number($("g-workers").value),retries:2,description:$("g-description").value,tags:[]
  };
}

function updatePreview(){
  try{const plan=buildPlan(),profiles=plan.request_formats.length*plan.context_modes.length,total=3*plan.samples_per_model*profiles;$("g-plan-preview").innerHTML=`<strong>总请求数 ${total}</strong><p>${profiles} 个格式格 · ${plan.temporal_windows} 个时间窗 · 每模型 ${plan.samples_per_model} 次 · 正式运行 ${plan.runtime_samples} 次/格</p>${plan.temporal_windows<3?'<p>单时间窗可以完成生成、分析和导入；当前版本将自定义探针作为参考证据。</p>':""}${plan.context_modes.includes("fixed_32k_history")?'<p class="cost-warning">包含固定 32K 历史。</p>':""}`;}
  catch(error){$("g-plan-preview").textContent=error.message;}
}

async function start(){
  try{const plan=buildPlan(),baseUrl=$("g-base-url").value.trim(),apiKey=$("g-key").value;if(!baseUrl||!apiKey||!plan.name||!plan.probe_id||!plan.user_prompt.trim())throw new Error("请完整填写可信连接和探针定义");gstate.plan=plan;await post("/api/generator/start",{base_url:baseUrl,api_key:apiKey,plan,resume_session_id:gstate.resumeSessionId});gstate.resumeSessionId=null;$("g-key").value="";$("g-resume-setup").classList.add("hidden");step("progress");watch();}
  catch(error){toast(error.message);}
}

function watch(){clearInterval(gstate.poller);gstate.poller=setInterval(pollStatus,1000);pollStatus();}
async function pollStatus(){
  try{const response=await fetch("/api/generator/status",{cache:"no-store"});if(!response.ok)return;const status=await response.json();renderProgress(status);if(["collected","complete","error","interrupted"].includes(status.status)){clearInterval(gstate.poller);gstate.poller=null;if(status.status==="collected")$("g-analyze").disabled=false;if(status.status==="interrupted")$("g-resume-setup").classList.remove("hidden");}}
  catch(_){/* next poll recovers */}
}

function renderProgress(status){
  const progress=status.progress||status,total=progress.planned||0,completed=progress.logical_completed||0;
  const label=({running:"正在采集可信三模型",collected:"采集完成，可以分析",complete:"分析与导出完成",error:"采集失败",stopping:"正在停止",interrupted:"进程曾中断；已完成任务保留，请重新填写 API key 后继续"})[status.status]||status.status;
  $("g-progress-text").textContent=status.error?`${label}：${status.error}`:label;
  $("g-progress-bar").style.width=`${total?Math.min(100,completed/total*100):status.status==="collected"?100:4}%`;
  $("g-progress-metrics").innerHTML=[["计划",total],["逻辑完成",completed],["成功",progress.successful||0],["最终错误",progress.errors||0],["HTTP尝试",progress.http_attempts||0],["重试",progress.retries||0],["剩余",progress.remaining??Math.max(0,total-(progress.successful||0))],["状态",status.status]].map(([key,value])=>`<div class="metric"><span>${key}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
}

async function analyze(){try{const result=await post("/api/generator/analyze",{});gstate.output=result;renderAnalysis(result.probe);step("analysis");}catch(error){toast(error.message);}}
async function useInDetector(){try{await post("/api/probe/use-generated",{});location.href="/";}catch(error){toast(error.message);}}

function renderAnalysis(probe){
  const baseline=probe.baseline_artifact||{},cells=baseline.cells||{},calibrations=baseline.calibrations||{};
  const diagnostics=baseline.reference_diagnostics?.profiles||{};
  const cellRows=Object.entries(cells).map(([key,value])=>`<tr><td>${escapeHtml(key)}</td><td>${Number(value.between_model_jsd||0).toFixed(3)}</td><td>${Number(value.within_model_jsd||0).toFixed(3)}</td><td>${Number(value.tau||0).toFixed(3)}</td><td>${Number(value.weight||0).toFixed(3)}</td></tr>`).join("")||'<tr><td colspan="5">当前数据无法计算统计单元，请查看下方采样完整性。</td></tr>';
  const calibrationRows=Object.values(calibrations).map(value=>`<div class="report-item"><strong>${escapeHtml(value.runtime_name)}</strong> · 正式 ${value.formal_eligible?"通过":"未通过"} · 回放 ${value.replay_count||0} · 错误上界 ${value.pass_metrics?.error_wilson95_upper===undefined?"—":(value.pass_metrics.error_wilson95_upper*100).toFixed(2)+"%"} · 总体/最差窗覆盖 ${((value.pass_metrics?.coverage_overall||0)*100).toFixed(1)}% / ${((value.pass_metrics?.coverage_worst_window||0)*100).toFixed(1)}%</div>`).join("");
  const diagnosticRows=Object.entries(diagnostics).flatMap(([profile,value])=>Object.entries(value.models||{}).map(([model,metrics])=>`<tr><td>${escapeHtml(profile)}</td><td>${escapeHtml(model.replace("gpt-5.6-",""))}</td><td>${metrics.windows??0}</td><td>${metrics.total??0}</td><td>${metrics.valid??0}</td><td>${((metrics.valid_rate||0)*100).toFixed(1)}%</td></tr>`)).join("")||'<tr><td colspan="6">旧版参考文件没有采样诊断；重新分析即可生成。</td></tr>';
  const allModels=Object.values(diagnostics).flatMap(value=>Object.values(value.models||{})),total=allModels.reduce((sum,value)=>sum+Number(value.total||0),0),valid=allModels.reduce((sum,value)=>sum+Number(value.valid||0),0),missing=[...new Set(Object.values(diagnostics).flatMap(value=>value.missing_models||[]))];
  const reason=baseline.reference_only_reason||"自定义探针参考分析已完成。";
  const incomplete=missing.length>0||baseline.reference_ready===false;
  $("g-analysis-result").innerHTML=`<div class="warning-banner ${probe.formal_eligible||!incomplete?"hidden":""}"><strong>采样不完整</strong><span>${escapeHtml(reason)}</span></div>${!probe.formal_eligible&&!incomplete?`<div class="report-item"><strong>单时间窗参考分析已完成</strong> · ${escapeHtml(reason)}</div>`:""}<div class="report-grid"><div class="metric"><span>已采样</span><strong>${total}</strong></div><div class="metric"><span>有效回答</span><strong>${valid}</strong></div><div class="metric"><span>缺失模型</span><strong>${escapeHtml(missing.map(value=>value.replace("gpt-5.6-","")).join("、")||"无")}</strong></div><div class="metric"><span>统计单元</span><strong>${Object.keys(cells).length}</strong></div></div>${calibrationRows}<h3>参考统计</h3><div class="data-table"><table><thead><tr><th>探针格</th><th>S</th><th>D</th><th>tau</th><th>w</th></tr></thead><tbody>${cellRows}</tbody></table></div><h3>采样完整性</h3><div class="data-table"><table><thead><tr><th>格式格</th><th>模型</th><th>窗口</th><th>采样</th><th>有效</th><th>有效率</th></tr></thead><tbody>${diagnosticRows}</tbody></table></div>`;
  $("g-output").textContent=probe.probe_identity.name;$("g-export-flags").textContent=`正式基线：${probe.formal_eligible?"是":"否"}；scoring=${baseline.scoring_version||"—"}；baseline SHA-256=${baseline.content_sha256||"—"}`;$("g-export-json").textContent=JSON.stringify(probe,null,2);
}

function applyPlan(plan){
  if(!plan)return;
  gstate.plan=plan;
  $("g-name").value=plan.name||"";$("g-probe-id").value=plan.probe_id||"";$("g-prompt").value=plan.user_prompt||"";$("g-description").value=plan.description||"";$("g-developer").value=plan.developer_prompt||"";
  $("g-sol").value=plan.model_names?.["gpt-5.6-sol"]||"gpt-5.6-sol";$("g-terra").value=plan.model_names?.["gpt-5.6-terra"]||"gpt-5.6-terra";$("g-luna").value=plan.model_names?.["gpt-5.6-luna"]||"gpt-5.6-luna";
  $("g-effort").value=plan.effort||"low";$("g-samples").value=plan.samples_per_model||100;$("g-runtime-samples").value=plan.runtime_samples||10;$("g-workers").value=plan.workers||20;$("g-normalizer").value=plan.normalizer?.id||"exact_trimmed_casefold";
  $("g-temporal").checked=Number(plan.temporal_windows||1)>1;$("g-gap").value=plan.window_gap_seconds??900;$("g-gap-row").classList.toggle("hidden",!$("g-temporal").checked);
  $("g-format-normal").checked=(plan.request_formats||[]).includes("normal");$("g-format-native").checked=(plan.request_formats||[]).includes("native_codex");$("g-context-none").checked=(plan.context_modes||[]).includes("no_history");$("g-context-32k").checked=(plan.context_modes||[]).includes("fixed_32k_history");updatePreview();
}

async function init(){
  const bootstrap=await fetch("/api/bootstrap").then(response=>response.json());gstate.token=bootstrap.session_token;
  document.querySelectorAll("[data-gstep]").forEach(node=>node.addEventListener("click",()=>step(node.dataset.gstep)));
  document.querySelectorAll("[data-next]").forEach(node=>node.addEventListener("click",()=>step(node.dataset.next)));
  document.querySelectorAll("[data-prev]").forEach(node=>node.addEventListener("click",()=>step(node.dataset.prev)));
  $("g-temporal").addEventListener("change",event=>{$("g-gap-row").classList.toggle("hidden",!event.target.checked);updatePreview();});
  ["g-samples","g-runtime-samples","g-workers","g-format-normal","g-format-native","g-context-none","g-context-32k"].forEach(id=>$(id).addEventListener("change",updatePreview));
  $("g-start").addEventListener("click",start);$("g-resume-setup").addEventListener("click",()=>step("trusted"));$("g-analyze").addEventListener("click",analyze);$("g-use-detector").addEventListener("click",useInDetector);updatePreview();
  const status=await fetch("/api/generator/status",{cache:"no-store"}).then(response=>response.json()).catch(()=>({status:"idle"}));
  if(status.resume_plan)applyPlan(status.resume_plan);
  if(["running","stopping"].includes(status.status)){step("progress");watch();}else if(status.status==="collected"){step("progress");renderProgress(status);$("g-analyze").disabled=false;}else if(status.status==="interrupted"){gstate.resumeSessionId=status.session_id;step("progress");renderProgress(status);$("g-resume-setup").classList.remove("hidden");}else if(status.status==="complete"&&status.output){step("progress");renderProgress(status);$("g-analyze").disabled=false;}
}
init().catch(error=>toast(`初始化失败：${error.message}`));
