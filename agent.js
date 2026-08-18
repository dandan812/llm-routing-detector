const state={token:"",profile:null};
const $=id=>document.getElementById(id);
const output=value=>{$("output").textContent=JSON.stringify(value,null,2);};
function profile(){return {product_tier:$("product-tier").value,client:$("client").value,protocol:$("protocol").value,context_mode:$("context-mode").value,baseline_version:$("baseline-version").value};}
function parse(id){try{return JSON.parse($(id).value);}catch(error){throw new Error(`${id} 不是有效 JSON`);}}
async function post(path,body){const response=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json","X-GPT56-Session":state.token},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw new Error(data.error||`HTTP ${response.status}`);return data;}
async function run(action){try{output(await action());}catch(error){output({error:String(error.message||error)});}}
async function boot(){const data=await (await fetch("/api/bootstrap")).json();state.token=data.session_token;const task={task_id:"nonce_roundtrip",required_tools:["probe.read_nonce","probe.write_result"],expected_final_state:{value:"ok",nonce_result:"42"},expected_nonce_result:"42"};$("task-json").value=JSON.stringify(task,null,2);$("tasks-json").value=JSON.stringify({nonce_roundtrip:task},null,2);}
$("score-agent").addEventListener("click",()=>run(()=>post("/api/agent/score",{profile:profile(),task:parse("task-json"),trace:parse("trace-json")})));
$("build-baseline").addEventListener("click",()=>run(()=>post("/api/agent/baseline",{profile:profile(),tasks:parse("tasks-json"),traces:parse("traces-json"),reference_model:$("reference-model").value})));
$("identify-agent").addEventListener("click",()=>run(()=>post("/api/agent/identify",{tasks:parse("tasks-json"),traces:parse("traces-json"),baselines:parse("baselines-json")})));
$("analyze-routing").addEventListener("click",()=>run(()=>post("/api/routing/analyze",{claimed_model:"gpt-5.6-sol",observations:parse("observations-json")})));
$("build-report").addEventListener("click",()=>run(()=>post("/api/three-layer/report",{api_fingerprint:parse("api-json"),agent_trajectory:parse("agent-json"),routing_drift:parse("drift-json")})));
boot().catch(error=>output({error:String(error.message||error)}));
