const state = {bootstrap:null, token:"", mode:"single", preset:"low", basePreset:"low", config:null, customProbes:[], poller:null, resumeSessionId:""};
const $ = id => document.getElementById(id);
const clone = value => JSON.parse(JSON.stringify(value));
const escapeHtml = value => String(value ?? "").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));

function toast(message) {
  const node=$("toast"); node.textContent=message; node.classList.add("show");
  clearTimeout(toast.timer); toast.timer=setTimeout(()=>node.classList.remove("show"),2600);
}

function showStep(name) {
  document.querySelectorAll(".page-step").forEach(node=>node.classList.toggle("active",node.id===`step-${name}`));
  document.querySelectorAll(".step").forEach(node=>node.classList.toggle("active",node.dataset.step===name));
  window.scrollTo({top:0,behavior:"smooth"});
}

function presetSource() { return state.mode==="single"?state.bootstrap.single_presets:state.bootstrap.continuous_presets; }

function applyPreset(name) {
  if(name==="custom") { markCustom("已切换到自定义档位"); return; }
  state.preset=name; state.basePreset=name; state.config=clone(presetSource()[name]);
  state.config.custom_probes=clone(state.customProbes); syncPlan();
}

function markCustom(reason) {
  if(state.preset!=="custom") state.basePreset=state.preset;
  state.preset="custom"; state.config.preset="custom"; state.config.base_preset=state.basePreset;
  state.config.custom_probes=clone(state.customProbes);
  $("custom-banner").classList.remove("hidden"); $("custom-diff").textContent=reason;
  document.querySelectorAll("[data-preset]").forEach(node=>node.classList.toggle("selected",node.dataset.preset==="custom"));
  updateEstimate();
}

function syncPlan() {
  document.querySelectorAll("[data-mode]").forEach(node=>node.classList.toggle("selected",node.dataset.mode===state.mode));
  document.querySelectorAll("[data-preset]").forEach(node=>node.classList.toggle("selected",node.dataset.preset===state.preset));
  $("custom-banner").classList.toggle("hidden",state.preset!=="custom");
  $("format-normal").checked=state.config.request_formats.includes("normal");
  $("format-native").checked=state.config.request_formats.includes("native_codex");
  $("context-none").checked=state.config.context_modes.includes("no_history");
  $("context-32k").checked=state.config.context_modes.includes("fixed_32k_history");
  $("workers").value=Number(state.config.workers??8);
  $("continuous-settings").classList.toggle("hidden",state.mode!=="continuous");
  if(state.mode==="continuous") {
    $("min-interval").value=Number(state.config.min_interval_seconds??150);
    $("max-interval").value=Number(state.config.max_interval_seconds??210);
    $("slots-per-cycle").value=Number(state.config.slots_per_cycle??1);
  }
  renderProbes(); updateEstimate();
}

function probeMeta(id) {
  const found=(state.bootstrap.probe_catalog||[]).find(item=>item.id===id);
  if(found) return found;
  if(id.startsWith("juice_")) return {name:id.replaceAll("_"," "),type:"Juice",description:"使用已冻结并通过可信筛选的模板池。"};
  if(id.startsWith("output_")) return {name:id.includes("48")?"Luna 48 输出控制":"Terra 32 输出控制",type:"防改写",description:"响应必须精确等于目标字面量。"};
  return {name:id,type:"探针",description:"自定义探测项目"};
}

function metricsText(meta) {
  if(!meta.between_model_jsd) return "";
  const format=items=>(items||[]).filter(value=>value!==null&&value!==undefined&&Number.isFinite(Number(value))).map(value=>Number(value).toFixed(3)).join(" / ")||"—";
  const error=meta.replay_error_wilson95_upper==null?"—":`${(Number(meta.replay_error_wilson95_upper)*100).toFixed(2)}%`;
  const coverage=meta.replay_coverage_overall==null?"—":`${(Number(meta.replay_coverage_overall)*100).toFixed(1)}% / ${(Number(meta.replay_coverage_worst_window)*100).toFixed(1)}%`;
  return `可信格式格 ${meta.trusted_profiles||0}；每模型可信窗口至少 ${meta.trusted_windows||0}；S ${format(meta.between_model_jsd)}；D ${format(meta.within_model_jsd)}；w ${format(meta.weights)}；最低有效率 ${((meta.minimum_valid_rate||0)*100).toFixed(1)}%；所属正式组合回放错误上界 ${error}，总体/最差窗覆盖 ${coverage}。`;
}

function customDocument(value){return value?.probe_file||value;}
function customRuntime(value,sourceJson=""){
  if(value?.probe_file_json||value?.probe_file)return value;
  const document=clone(value),enabled=value?.enabled??true,probability=Number(value?.probability_percent??100),windowSize=Number(value?.window??20),requests=Number(value?.runtime_requests??10);
  delete document.enabled;delete document.probability_percent;delete document.window;
  return {probe_file:document,probe_file_json:sourceJson||JSON.stringify(document),enabled,runtime_requests:requests,probability_percent:probability,window:windowSize};
}
function probeSetting(probe,isCustom=false){
  return state.mode==="single"?`${Number(isCustom?probe.runtime_requests:probe.requests??0)} 次${isCustom?" / 格":""}`:`${Number(probe.probability_percent??0)}% · 窗口 ${Number(probe.window??20)}`;
}
function updateProbeSetting(row,probe,isCustom=false){row.querySelector(".probe-setting").textContent=probeSetting(probe,isCustom);}

function customMetrics(probe){
  probe=customDocument(probe);
  const artifact=probe.baseline_artifact||{},cells=Object.values(artifact.cells||{}),calibrations=Object.values(artifact.calibrations||{}).filter(value=>value.formal_eligible),pass=calibrations.map(value=>value.pass_metrics||{});
  const format=values=>values.filter(value=>value!=null&&Number.isFinite(Number(value))).map(value=>Number(value).toFixed(3)).join(" / ")||"—";
  const error=pass.length?Math.max(...pass.map(value=>Number(value.error_wilson95_upper??1))):null;
  const overall=pass.length?Math.min(...pass.map(value=>Number(value.coverage_overall??0))):null;
  const worst=pass.length?Math.min(...pass.map(value=>Number(value.coverage_worst_window??0))):null;
  return `S ${format(cells.map(value=>value.between_model_jsd))}；D ${format(cells.map(value=>value.within_model_jsd))}；tau ${format(cells.map(value=>value.tau))}；w ${format(cells.map(value=>value.weight))}；回放错误上界 ${error==null?"—":(error*100).toFixed(2)+"%"}；总体/最差窗覆盖 ${overall==null?"—":(overall*100).toFixed(1)+"% / "+(worst*100).toFixed(1)+"%"}。`;
}

function renderProbes() {
  const list=$("probe-list"); list.textContent="";
  Object.entries(state.config.probes).forEach(([id,probe])=>{
    const meta=probeMeta(id), row=document.createElement("div"); row.className="probe-row";
    const setting=probeSetting(probe);
    const controls=state.mode==="single"
      ? `<label>请求数<input class="probe-requests" type="number" min="0" value="${Number(probe.requests??0)}"></label>`
      : `<label>每槽概率 %<input class="probe-probability" type="number" min="0" max="100" value="${Number(probe.probability_percent??0)}"></label><label>滚动窗口<input class="probe-window" type="number" min="1" value="${Number(probe.window??20)}"></label>`;
    row.innerHTML=`<div class="probe-summary"><input class="probe-enabled" type="checkbox" ${probe.enabled?"checked":""}><span class="probe-name">${escapeHtml(meta.name)}</span><span class="probe-type">${escapeHtml(meta.type)}</span><span class="probe-setting">${escapeHtml(setting)}</span><button class="expand-probe" title="展开详情" aria-label="展开详情">▸</button></div><div class="probe-details"><p>${escapeHtml(meta.description)}</p>${metricsText(meta)?`<p><small>${escapeHtml(metricsText(meta))}</small></p>`:""}<div class="probe-fields">${controls}<label>思考强度<input value="${escapeHtml(probe.effort??"固定")}" disabled></label></div></div>`;
    row.querySelector(".expand-probe").addEventListener("click",()=>row.classList.toggle("open"));
    row.querySelector(".probe-enabled").addEventListener("change",event=>{probe.enabled=event.target.checked;updateProbeSetting(row,probe);markCustom(`${meta.name} 启用状态已修改`);});
    row.querySelector(".probe-requests")?.addEventListener("input",event=>{probe.requests=Number(event.target.value);updateProbeSetting(row,probe);markCustom(`${meta.name} 请求数已修改`);});
    row.querySelector(".probe-probability")?.addEventListener("input",event=>{probe.probability_percent=Number(event.target.value);updateProbeSetting(row,probe);markCustom(`${meta.name} 持续概率已修改`);});
    row.querySelector(".probe-window")?.addEventListener("input",event=>{probe.window=Number(event.target.value);updateProbeSetting(row,probe);markCustom(`${meta.name} 窗口已修改`);});
    list.appendChild(row);
  });
  state.customProbes=state.customProbes.map(customRuntime);
  state.customProbes.forEach((probe,index)=>{
    const row=document.createElement("div"); row.className="probe-row";
    probe.enabled??=true;probe.runtime_requests??=10; probe.probability_percent??=100; probe.window??=20;
    const probeFile=customDocument(probe),artifact=probeFile.baseline_artifact||{};
    const probeId=probeFile.probe_identity.probe_id,profiles=Object.keys(artifact.raw_counts?.[probeId]?.profiles||{});
    const setting=probeSetting(probe,true);
    const controls=state.mode==="single"
      ? `<label>请求数 / 格<input class="custom-requests" type="number" min="1" value="${Number(probe.runtime_requests)}"></label>`
      : `<label>每槽概率 %<input class="custom-probability" type="number" min="0" max="100" value="${Number(probe.probability_percent)}"></label><label>滚动窗口<input class="custom-window" type="number" min="1" value="${Number(probe.window)}"></label>`;
    row.innerHTML=`<div class="probe-summary"><input class="custom-enabled" type="checkbox" ${probe.enabled?"checked":""}><span class="probe-name">${escapeHtml(probeFile.probe_identity.name)}</span><span class="probe-type">自定义概率（参考）</span><span class="probe-setting">${escapeHtml(setting)}</span><button class="expand-probe" title="展开详情" aria-label="展开详情">▸</button></div><div class="probe-details"><p>${escapeHtml(probeFile.exact_prompts_and_hashes.user_prompt)}</p><p><small>适用格式格：${escapeHtml(profiles.join("、")||"无")}；参考数据：${probeFile.reference_ready||probeFile.formal_eligible?"可用":"采样不完整"}；本版本导入后固定为参考证据，不参与正式硬结论；${escapeHtml(customMetrics(probeFile))}</small></p><div class="probe-fields">${controls}<button class="remove-probe danger" title="移除此探针" aria-label="移除此探针">×</button></div></div>`;
    row.querySelector(".expand-probe").addEventListener("click",()=>row.classList.toggle("open"));
    row.querySelector(".custom-enabled").addEventListener("change",event=>{probe.enabled=event.target.checked;updateProbeSetting(row,probe,true);markCustom(`${probeFile.probe_identity.name} 启用状态已修改`);});
    row.querySelector(".custom-requests")?.addEventListener("input",event=>{probe.runtime_requests=Number(event.target.value);updateProbeSetting(row,probe,true);markCustom(`${probeFile.probe_identity.name} 请求数已修改`);});
    row.querySelector(".custom-probability")?.addEventListener("input",event=>{probe.probability_percent=Number(event.target.value);updateProbeSetting(row,probe,true);markCustom(`${probeFile.probe_identity.name} 持续概率已修改`);});
    row.querySelector(".custom-window")?.addEventListener("input",event=>{probe.window=Number(event.target.value);updateProbeSetting(row,probe,true);markCustom(`${probeFile.probe_identity.name} 窗口已修改`);});
    row.querySelector(".remove-probe").addEventListener("click",()=>{state.customProbes.splice(index,1);state.config.custom_probes=clone(state.customProbes);markCustom(`${probeFile.probe_identity.name} 已移除`);renderProbes();});
    list.appendChild(row);
  });
}

async function post(path,body) {
  const response=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json","X-GPT56-Session":state.token},body:JSON.stringify(body)});
  const value=await response.json(); if(!response.ok) throw new Error(value.error||`HTTP ${response.status}`); return value;
}

async function updateEstimate() {
  try {
    const estimate=await post("/api/detector/estimate",{config:state.config});
    const items=state.mode==="single"
      ? [["请求数",estimate.total_requests],["32K 请求",estimate.fixed_32k_requests],["长上下文词量",estimate.approximate_fixed_32k_input_tokens?.toLocaleString("zh-CN")]]
      : [["格式格",estimate.profiles],["调度","随机间隔 · 独立概率"],["档位",estimate.official?"官方":"自定义参考"]];
    $("estimate").innerHTML=items.map(([key,value])=>`<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value??"—")}</dd></div>`).join("");
    const costly=state.mode==="single"&&estimate.fixed_32k_requests>0;
    $("high-cost").classList.toggle("hidden",!costly); if(costly) $("high-cost").textContent=`当前含 ${estimate.fixed_32k_requests} 条固定 32K 请求。`;
  } catch(error) { console.warn(error); }
}

function bindCollection(id,collection,value) {
  $(id).addEventListener("change",event=>{
    const values=state.config[collection];
    if(event.target.checked&&!values.includes(value)) values.push(value);
    if(!event.target.checked) state.config[collection]=values.filter(item=>item!==value);
    if(!state.config[collection].length){event.target.checked=true;state.config[collection].push(value);toast("至少保留一个选项");return;}
    markCustom("请求格式或上下文已修改");
  });
}

async function startRun() {
  const baseUrl=$("base-url").value.trim(),apiKey=$("api-key").value,model=$("model").value.trim();
  if(!baseUrl||!apiKey||!model){toast("请完整填写 API 地址、模型名和 key");showStep("connect");return;}
  const retention=$("retention-enabled").checked,retentionPath=$("retention-path").value.trim();
  if(retention&&!retentionPath){toast("开启留存后必须填写绝对目录");return;}
  try {
    const result=await post("/api/detector/start",{base_url:baseUrl,model,api_key:apiKey,config:state.config,resume_session_id:state.resumeSessionId||null,retention_enabled:retention,retention_directory:retentionPath||null});
    state.resumeSessionId=""; $("api-key").value=""; $("run-session").textContent=`会话 ${result.session_id}`; showStep("run"); watchStatus();
  } catch(error){toast(error.message);}
}

function watchStatus() {
  clearInterval(state.poller); state.poller=setInterval(pollStatus,1000); pollStatus();
}

async function pollStatus() {
  try {
    const response=await fetch("/api/detector/status",{cache:"no-store"}); if(!response.ok)return;
    const status=await response.json(); renderRunStatus(status);
    if(["complete","stopped","error"].includes(status.status)){clearInterval(state.poller);state.poller=null;if(status.report_available)loadReport();}
  } catch(_) { /* next one-second poll recovers */ }
}

function renderRunStatus(status) {
  $("raw-status").textContent=JSON.stringify(status,null,2);
  if(status.session_id) $("run-session").textContent=`会话 ${status.session_id}`;
  const statusLabel=({running:"运行中",complete:"已完成",error:"运行错误",stopping:"正在停止",stopped:"已停止",interrupted:"进程中断，可输入 key 恢复"})[status.status]||status.status;
  $("run-status").textContent=status.error?`${statusLabel}：${status.error}`:statusLabel;
  $("run-status").className=`status-dot ${status.status}`; $("run-updated").textContent=status.updated_at||"—";
  const progress=status.progress||{},total=progress.planned||0,done=progress.logical_completed||0;
  $("run-progress").style.width=`${total?Math.min(100,done/total*100):status.status==="complete"?100:5}%`;
  $("run-details").innerHTML=[["逻辑完成",done],["成功",progress.successful??0],["最终错误",progress.errors??0],["取消",progress.cancelled??0],["HTTP尝试",progress.http_attempts??0],["重试",progress.retries??0],["在途",progress.in_flight??0],["计划",total],["结论",status.verdict||"计算中"]].map(([key,value])=>`<div class="metric"><span>${key}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
}

async function loadReport() {
  const response=await fetch("/api/detector/report",{cache:"no-store"}); if(!response.ok)return;
  renderReport(await response.json()); showStep("report");
}

function renderReport(report) {
  $("report-empty").classList.add("hidden"); $("report-content").classList.remove("hidden");
  $("verdict-title").textContent=report.title_cn||report.overall_verdict||"未形成正式结论";
  $("verdict-subtitle").textContent=report.subtitle_cn||report.quality_note||(report.common_causes?.length?`常见原因：${report.common_causes.join("；")}`:"");
  const alert=["Juice混用","仅概率探针混用","可能非GPT"].includes(report.overall_verdict);
  $("verdict-band").classList.toggle("alert",alert); $("verdict-band").classList.toggle("warning",!report.verdict_available||String(report.overall_verdict||"").includes("证据不足"));
  $("report-custom").classList.toggle("hidden",!report.custom_preset);
  if(report.custom_preset) $("report-custom").querySelector("span").textContent=`修改字段：${(report.custom_changes||[]).join("、")||"自定义探针"}；official=false。`;
  const probability=report.probability_summary||{},calibration=probability.calibration||{},passMetrics=calibration.pass_metrics||{},mixture=probability.mixture||{};
  const mixtureShares=Object.entries(mixture.proportions||{}).map(([models,share])=>`${models.replaceAll("gpt-5.6-","")} ${(Number(share)*100).toFixed(1)}%`).join(" / ")||"—";
  const completeness=report.data_completeness||{},probabilityComplete=completeness.probability||{};
  $("evidence-summary").innerHTML=[
    ["可信回放错误率上界",passMetrics.error_wilson95_upper===undefined?"—":`${(Number(passMetrics.error_wilson95_upper)*100).toFixed(2)}%`],
    ["回放总体/最差窗覆盖",passMetrics.coverage_overall===undefined?"—":`${(Number(passMetrics.coverage_overall)*100).toFixed(1)}% / ${(Number(passMetrics.coverage_worst_window)*100).toFixed(1)}%`],
    ["混合比例",mixtureShares],
    ["混合 gain / 门槛",mixture.mixture_gain===undefined?"—":`${Number(mixture.mixture_gain).toFixed(4)} / ${mixture.mixture_gain_threshold===null||mixture.mixture_gain_threshold===undefined?"—":Number(mixture.mixture_gain_threshold).toFixed(4)}`],
    ["混合可辨识",mixture.identifiable===undefined?"—":mixture.identifiable?"可分别估计":"含不可区分模型组"],
    ["数据完整性",probabilityComplete.enabled?(probabilityComplete.formal_eligible?"正式样本完整":"不足："+(probabilityComplete.reasons||[]).join("；")):"未启用概率探针"],
  ].map(([key,value])=>`<div class="metric"><span>${escapeHtml(key)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
  const failures=(report.failed_items||[]).map(item=>`<div class="report-item"><strong>${escapeHtml(item.layer)}</strong> · ${escapeHtml(item.reason)}</div>`).join("")||'<div class="report-item">没有确定性失败项目。</div>';
  const errors=(report.network_error_details||[]).map(item=>`<div class="report-item error-item"><strong>${escapeHtml(item.probe_id||"请求")}</strong> · ${escapeHtml(item.stage)}/${escapeHtml(item.category)} · HTTP ${escapeHtml(item.http_status??"—")} · 第 ${escapeHtml(item.attempt)} 次 · ${escapeHtml(item.safe_message)}</div>`).join("");
  $("failed-items").innerHTML=failures+errors;
  $("probability-bars").innerHTML=(report.probability_details||[]).map(item=>`<div class="probability-row"><strong>${escapeHtml(item.label_cn)}</strong><div class="bar"><span style="width:${Math.max(0,Math.min(100,item.probability*100))}%"></span></div><span class="probability-value">${(item.probability*100).toFixed(3)}%</span></div>`).join("")||'<div class="report-item">未启用或尚无正式概率证据。</div>';
  const efforts=Object.entries(report.juice_summary?.per_effort||{});
  $("juice-result").innerHTML=`<table><thead><tr><th>档位</th><th>尝试</th><th>有效</th><th>当前成功</th><th>混用</th><th>不成功</th><th>网络错误</th><th>共享成功</th></tr></thead><tbody>${efforts.map(([effort,value])=>`<tr><td>${effort}</td><td>${value.attempted}</td><td>${value.valid_completed}</td><td>${value.current_success}</td><td>${value.mixed}</td><td>${value.unsuccessful}</td><td>${value.network_error}</td><td>${value.shared_current_success}</td></tr>`).join("")}</tbody></table>`;
  const cells=report.probability_summary?.cell_details||{},families=report.probability_summary?.family_contributions||{};
  $("probe-results").innerHTML=`<table><thead><tr><th>探针格</th><th>实际/要求</th><th>答案分布</th><th>S / D / tau / w</th><th>Sol/Terra/Luna 局部似然</th><th>家族最终贡献</th><th>OOD</th></tr></thead><tbody>${Object.entries(cells).map(([key,value])=>{const family=families[value.probe_id]||{};return `<tr><td>${escapeHtml(key)}</td><td>${value.sample_count}/${value.required_samples}</td><td>${escapeHtml(JSON.stringify(value.counts))}</td><td>${Number(value.between_model_jsd).toFixed(3)} / ${Number(value.within_model_jsd).toFixed(3)} / ${Number(value.tau).toFixed(3)} / ${Number(value.weight).toFixed(3)}</td><td>${escapeHtml(JSON.stringify(value.average_log_likelihood))}</td><td>${escapeHtml(JSON.stringify(family.model_contributions||{}))}</td><td>${value.isolated_ood?escapeHtml(value.ood_reason):"否"}</td></tr>`;}).join("")}</tbody></table>`;
  $("profile-results").innerHTML=`<table><thead><tr><th>格式格</th><th>任务</th><th>成功</th><th>错误</th><th>取消</th></tr></thead><tbody>${Object.entries(report.profile_summary||{}).map(([profile,value])=>`<tr><td>${escapeHtml(profile)}</td><td>${value.logical_tasks}</td><td>${value.successful}</td><td>${value.final_errors}</td><td>${value.cancelled}</td></tr>`).join("")}</tbody></table>`;
  const network=report.network_summary||{};
  $("network-result").innerHTML=[["逻辑任务",network.logical_tasks],["逻辑完成",network.logical_completed],["成功",network.successful],["最终错误",network.final_errors],["取消",network.cancelled],["HTTP尝试",network.http_attempts],["重试",network.retries]].map(([key,value])=>`<div class="metric"><span>${key}</span><strong>${value??0}</strong></div>`).join("");
  $("limitations").innerHTML=(report.limitations||[]).map(item=>`<p>• ${escapeHtml(item)}</p>`).join("");
  $("report-json").textContent=JSON.stringify(report,null,2);
}

async function init() {
  state.bootstrap=await fetch("/api/bootstrap").then(response=>response.json()); state.token=state.bootstrap.session_token; applyPreset("low");
  if(state.bootstrap.pending_custom_probe){const pending=customRuntime(clone(state.bootstrap.pending_custom_probe));state.customProbes.push(pending);state.config.custom_probes=clone(state.customProbes);markCustom(`已加入 ${customDocument(pending).probe_identity.name}`);renderProbes();}
  document.querySelectorAll(".step").forEach(node=>node.addEventListener("click",()=>node.dataset.step==="report"?loadReport():showStep(node.dataset.step)));
  document.querySelectorAll("[data-mode]").forEach(node=>node.addEventListener("click",()=>{state.mode=node.dataset.mode;applyPreset(state.basePreset);}));
  document.querySelectorAll("[data-preset]").forEach(node=>node.addEventListener("click",()=>applyPreset(node.dataset.preset)));
  $("to-plan").addEventListener("click",()=>showStep("plan")); $("back-connect").addEventListener("click",()=>showStep("connect"));
  $("restore-defaults").addEventListener("click",()=>{if(confirm("恢复本档默认参数会覆盖当前修改，是否继续？"))applyPreset(state.basePreset);});
  $("start-run").addEventListener("click",startRun); $("stop-run").addEventListener("click",async()=>{try{await post("/api/detector/stop",{});toast("正在停止");}catch(error){toast(error.message);}});
  $("new-session").addEventListener("click",()=>{state.resumeSessionId="";showStep("connect");});
  bindCollection("format-normal","request_formats","normal"); bindCollection("format-native","request_formats","native_codex"); bindCollection("context-none","context_modes","no_history"); bindCollection("context-32k","context_modes","fixed_32k_history");
  $("workers").addEventListener("change",event=>{state.config.workers=Number(event.target.value);markCustom("并发数已修改");});
  [["min-interval","min_interval_seconds"],["max-interval","max_interval_seconds"],["slots-per-cycle","slots_per_cycle"]].forEach(([id,key])=>$(id).addEventListener("change",event=>{state.config[key]=Number(event.target.value);markCustom("持续调度参数已修改");}));
  $("retention-enabled").addEventListener("change",event=>$("retention-path-row").classList.toggle("hidden",!event.target.checked));
  $("probe-file").addEventListener("change",async event=>{const file=event.target.files[0];if(!file)return;try{const sourceJson=await file.text(),probe=JSON.parse(sourceJson);await post("/api/probe/verify",{probe_file_json:sourceJson});state.customProbes.push(customRuntime(probe,sourceJson));state.config.custom_probes=clone(state.customProbes);markCustom(`已导入 ${probe.probe_identity.name}`);renderProbes();}catch(error){toast(error.message);}});
  const current=await fetch("/api/detector/status",{cache:"no-store"}).then(response=>response.json()).catch(()=>({status:"idle"}));
  if(["running","stopping"].includes(current.status)){showStep("run");watchStatus();}
  else if(["complete","stopped"].includes(current.status)&&current.report_available){await loadReport();}
  else if(current.status==="interrupted"){
    state.resumeSessionId=current.session_id||"";
    if(current.resume_config){state.config=clone(current.resume_config);state.mode=state.config.mode;state.preset=state.config.preset;state.basePreset=state.config.base_preset||state.preset;state.customProbes=clone(state.config.custom_probes||[]);syncPlan();}
    if(current.claimed_model) $("model").value=current.claimed_model;
    if(current.safe_endpoint) $("base-url").value=current.safe_endpoint;
    toast("检测进程曾中断；输入 API key 后会按原冻结任务和剩余尝试预算恢复");
    showStep("connect");
  }
}

init().catch(error=>toast(`初始化失败：${error.message}`));
