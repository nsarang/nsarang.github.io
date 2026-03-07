var H=Object.defineProperty;var C=e=>{throw TypeError(e)};var j=(e,r,o)=>r in e?H(e,r,{enumerable:!0,configurable:!0,writable:!0,value:o}):e[r]=o;var M=(e,r,o)=>j(e,typeof r!="symbol"?r+"":r,o),X=(e,r,o)=>r.has(e)||C("Cannot "+o);var $=(e,r,o)=>(X(e,r,"read from private field"),o?o.call(e):r.get(e)),y=(e,r,o)=>r.has(e)?C("Cannot add the same private member more than once"):r instanceof WeakSet?r.add(e):r.set(e,o),U=(e,r,o,t)=>(X(e,r,"write to private field"),t?t.call(e,o):r.set(e,o),o);var G=(e,r,o,t)=>({set _(n){U(e,r,n,o)},get _(){return $(e,r,t)}});import{S as b,E as k,U as V,d as q,D as E,s as d,A as c,r as Y,i as L,b as P,c as J}from"./index.js";import"./vendor.js";const Q=`
uvec2 threefry2x32(uvec2 key, uvec2 ctr) {
  uint ks0 = key.x;
  uint ks1 = key.y;
  uint ks2 = ks0 ^ ks1 ^ 0x1BD11BDAu;

  uint x0 = ctr.x + ks0;
  uint x1 = ctr.y + ks1;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks1;
  x1 += ks2 + 1u;

  x0 += x1; x1 = (x1 << 17u) | (x1 >> 15u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 29u) | (x1 >> 3u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 16u) | (x1 >> 16u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 24u) | (x1 >> 8u); x1 ^= x0;
  x0 += ks2;
  x1 += ks0 + 2u;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks0;
  x1 += ks1 + 3u;

  x0 += x1; x1 = (x1 << 17u) | (x1 >> 15u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 29u) | (x1 >> 3u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 16u) | (x1 >> 16u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 24u) | (x1 >> 8u); x1 ^= x0;
  x0 += ks1;
  x1 += ks2 + 4u;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks2;
  x1 += ks0 + 5u;

  return uvec2(x0, x1);
}`,Z=`
const float _erf_p = 0.3275911;
const float _erf_a1 = 0.254829592;
const float _erf_a2 = -0.284496736;
const float _erf_a3 = 1.421413741;
const float _erf_a4 = -1.453152027;
const float _erf_a5 = 1.061405429;
float erf(float x) {
  float t = 1.0 / (1.0 + _erf_p * abs(x));
  float P_t = (((((_erf_a5 * t) + _erf_a4) * t + _erf_a3) * t + _erf_a2) * t + _erf_a1) * t;
  return sign(x) * (1.0 - P_t * exp(-x * x));
}
float erfc(float x) {
  float t = 1.0 / (1.0 + _erf_p * abs(x));
  float P_t = (((((_erf_a5 * t) + _erf_a4) * t + _erf_a3) * t + _erf_a2) * t + _erf_a1) * t;
  float E = P_t * exp(-x * x);
  return x >= 0.0 ? E : 2.0 - E;
}`;var A,F,I,B,z,ce=(z=class{constructor(e){M(this,"type","webgl");M(this,"maxArgs",8);M(this,"gl");y(this,A);y(this,F);y(this,I);y(this,B);this.gl=e,U(this,A,e.createFramebuffer()),U(this,F,new Map),U(this,I,new Map),U(this,B,1)}malloc(e,r){const o=this.gl,t=Math.ceil(e/4)||1,n=Math.ceil(t/4)||1,{width:_,height:a}=K(n),x=o.createTexture();if(!x)throw new Error("Failed to create texture");o.bindTexture(o.TEXTURE_2D,x),o.texParameteri(o.TEXTURE_2D,o.TEXTURE_MIN_FILTER,o.NEAREST),o.texParameteri(o.TEXTURE_2D,o.TEXTURE_MAG_FILTER,o.NEAREST),o.texParameteri(o.TEXTURE_2D,o.TEXTURE_WRAP_S,o.CLAMP_TO_EDGE),o.texParameteri(o.TEXTURE_2D,o.TEXTURE_WRAP_T,o.CLAMP_TO_EDGE);const u=_*a*4;let l=null;r&&(l=new Float32Array(u),new Uint8Array(l.buffer).set(r)),o.texImage2D(o.TEXTURE_2D,0,o.RGBA32F,_,a,0,o.RGBA,o.FLOAT,l),o.bindTexture(o.TEXTURE_2D,null);const i=G(this,B)._++;return $(this,F).set(i,{ref:1,size:e,texture:x,width:_,height:a}),i}incRef(e){const r=$(this,F).get(e);if(!r)throw new b(e);r.ref++}decRef(e){const r=$(this,F).get(e);if(!r)throw new b(e);r.ref--,r.ref===0&&(this.gl.deleteTexture(r.texture),$(this,F).delete(e))}async read(e,r,o){const t=$(this,F).get(e);if(!t)throw new b(e);const n=this.gl;r===void 0&&(r=0),o===void 0&&(o=t.size-r),n.bindFramebuffer(n.FRAMEBUFFER,$(this,A)),n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,t.texture,0);const _=t.width*t.height*4*4,a=new Float32Array(_/4),x=n.createBuffer();if(!x)throw new Error("Failed to create PBO");n.bindBuffer(n.PIXEL_PACK_BUFFER,x),n.bufferData(n.PIXEL_PACK_BUFFER,_,n.STREAM_READ),n.readPixels(0,0,t.width,t.height,n.RGBA,n.FLOAT,0);const u=n.getError();if(u!==n.NO_ERROR)throw n.deleteBuffer(x),new Error(`WebGL error after readPixels: ${u}`);const l=n.fenceSync(n.SYNC_GPU_COMMANDS_COMPLETE,0);if(!l)throw new Error("Failed to create sync object");n.flush(),n.bindBuffer(n.PIXEL_PACK_BUFFER,null),n.bindFramebuffer(n.FRAMEBUFFER,null),await new Promise((s,f)=>{const T=()=>{const h=n.clientWaitSync(l,0,0);if(h===n.TIMEOUT_EXPIRED){setTimeout(T,5);return}if(h===n.WAIT_FAILED){n.deleteSync(l),n.deleteBuffer(x),f(new Error("clientWaitSync failed"));return}s()};T()}),n.deleteSync(l),n.bindBuffer(n.PIXEL_PACK_BUFFER,x),n.getBufferSubData(n.PIXEL_PACK_BUFFER,0,a),n.bindBuffer(n.PIXEL_PACK_BUFFER,null),n.deleteBuffer(x);const i=new Uint8Array(a.buffer);return new Uint8Array(i.slice(r,r+o))}readSync(e,r,o){const t=$(this,F).get(e);if(!t)throw new b(e);const n=this.gl;r===void 0&&(r=0),o===void 0&&(o=t.size-r),n.bindFramebuffer(n.FRAMEBUFFER,$(this,A)),n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,t.texture,0);const _=t.width*t.height*4,a=new Float32Array(_);n.readPixels(0,0,t.width,t.height,n.RGBA,n.FLOAT,a),n.bindFramebuffer(n.FRAMEBUFFER,null);const x=new Uint8Array(a.buffer);return new Uint8Array(x.slice(r,r+o))}async prepareKernel(e){return this.prepareKernelSync(e)}prepareKernelSync(e){const r=ee(e),o=$(this,I).get(r.code);if(o)return new k(e,o);const t=oe(this.gl,r);return $(this,I).set(r.code,t),new k(e,t)}prepareRoutine(e){throw new V(e.name,"webgl")}prepareRoutineSync(e){throw new V(e.name,"webgl")}dispatch(e,r,o){const t=this.gl;if(t.isContextLost())throw new Error("WebGL context lost - cannot dispatch");const{program:n,inputLocations:_}=e.data;if(r.length!==e.data.numInputs)throw new Error(`Expected ${e.data.numInputs} inputs, got ${r.length}`);if(o.length!==1)throw new Error(`Expected 1 output, got ${o.length}`);const a=$(this,F).get(o[0]);if(!a)throw new b(o[0]);t.bindFramebuffer(t.FRAMEBUFFER,$(this,A)),t.framebufferTexture2D(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0,t.TEXTURE_2D,a.texture,0);const x=t.checkFramebufferStatus(t.FRAMEBUFFER);if(x!==t.FRAMEBUFFER_COMPLETE)throw new Error(`Framebuffer incomplete: ${x}`);t.viewport(0,0,a.width,a.height),t.useProgram(n);for(let l=0;l<r.length;l++){const i=$(this,F).get(r[l]);if(!i)throw new b(r[l]);t.activeTexture(t.TEXTURE0+l),t.bindTexture(t.TEXTURE_2D,i.texture),_[l]!==null&&t.uniform1i(_[l],l)}t.drawArrays(t.TRIANGLES,0,3);const u=t.getError();if(u!==t.NO_ERROR){let l;throw u===t.INVALID_ENUM?l="INVALID_ENUM":u===t.INVALID_VALUE?l="INVALID_VALUE":u===t.INVALID_OPERATION?l="INVALID_OPERATION":u===t.INVALID_FRAMEBUFFER_OPERATION?l="INVALID_FRAMEBUFFER_OPERATION":u===t.OUT_OF_MEMORY?l="OUT_OF_MEMORY":u===t.CONTEXT_LOST_WEBGL?l="CONTEXT_LOST_WEBGL":l=`UNKNOWN(${u})`,new Error(`WebGL error after drawArrays: ${l}`)}t.bindFramebuffer(t.FRAMEBUFFER,null),t.useProgram(null)}},A=new WeakMap,F=new WeakMap,I=new WeakMap,B=new WeakMap,z);function ee(e){var v;const r=q(e),{nargs:o,reduction:t}=e,n=e.dtype,_=Math.ceil(e.size/4)||1,a=K(_),x=Array(o).fill(E.Float32),u={erf:!1,threefry:!1},l=p=>{p.op===c.GlobalIndex?x[p.arg[0]]=p.dtype:p.op===c.Erf||p.op===c.Erfc?u.erf=!0:p.op===c.Threefry2x32&&(u.threefry=!0)};r.exp.fold(l),(v=r.epilogue)==null||v.fold(l);const i=[];let s="";const f=Symbol("pushIndent"),T=Symbol("popIndent"),h=(...p)=>{for(const w of p)w===f?s+="  ":w===T?s=s.slice(0,-2):i.push(w&&s+w)};h("#version 300 es","precision highp float;","precision highp int;","");const R=Array.from({length:o},(p,w)=>`in${w}`),m=D(n);for(let p=0;p<o;p++)h(`uniform highp sampler2D ${R[p]};`);h("out vec4 out0;");const S=new Set;for(const p of x)S.add(p);for(const p of S)h(ne(p));if(u.erf&&h(Z),u.threefry&&h(Q),h(`${m} compute(int gidx) {`,f,`${m} result = ${N(n,0)};`,`if (gidx < ${e.size}) {`,f),t){const p=D(t.dtype),w=N(t.dtype,t.identity);h(`${p} acc = ${w};`,`for (int ridx = 0; ridx < ${r.size.reduce}; ridx++) {`,f);const g=O(r.exp,R,x);if(t.op===c.Add)h(`acc += ${d(g)};`);else if(t.op===c.Mul)h(`acc *= ${d(g)};`);else if(t.op===c.Min)t.dtype!==E.Bool?h(`acc = min(acc, ${d(g)});`):h(`acc = acc && ${g};`);else if(t.op===c.Max)t.dtype!==E.Bool?h(`acc = max(acc, ${d(g)});`):h(`acc = acc || ${g};`);else throw new Error(`Unsupported reduction op: ${t.op}`);h(T,"}"),h(`result = ${O(r.epilogue,R,x)};`)}else{const p=O(r.exp,R,x);h(`result = ${d(p)};`)}return h(T,"}","return result;",T,`}
`),h("void main() {",f,"ivec2 fragCoord = ivec2(gl_FragCoord.xy);",`int texelIdx = fragCoord.y * ${a.width} + fragCoord.x;`,`${m} result0 = compute(texelIdx * 4);`,`${m} result1 = compute(texelIdx * 4 + 1);`,`${m} result2 = compute(texelIdx * 4 + 2);`,`${m} result3 = compute(texelIdx * 4 + 3);`,`out0 = vec4(${Y(4).map(p=>ie(n,`result${p}`)).join(", ")});`),h(T,"}"),{code:i.join(`
`),numInputs:o,outputSize:[a.width,a.height],outputDtype:n}}function W(e,r,o){const t=e.createShader(r);if(e.shaderSource(t,o),e.compileShader(t),!e.getShaderParameter(t,e.COMPILE_STATUS))throw new Error(e.getShaderInfoLog(t)??"Unknown shader compile error");return t}function te(e,r,o){const t=e.createProgram();if(e.attachShader(t,W(e,e.VERTEX_SHADER,r)),e.attachShader(t,W(e,e.FRAGMENT_SHADER,o)),e.linkProgram(t),!e.getProgramParameter(t,e.LINK_STATUS))throw new Error(e.getProgramInfoLog(t)??"Unknown program link error");return t}const re=`#version 300 es
precision highp float;
const vec2 pos[3] = vec2[](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
void main() { gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0); }
`;function oe(e,r){const o=te(e,re,r.code),t=[];for(let n=0;n<r.numInputs;n++)t.push(e.getUniformLocation(o,`in${n}`));return{...r,program:o,inputLocations:t}}function K(e){let o=Math.min(Math.ceil(Math.sqrt(e)),16384);o=Math.min(1<<Math.ceil(Math.log2(o)),16384);const t=Math.min(Math.ceil(e/o),16384);return{width:o,height:t}}function D(e){switch(e){case E.Float32:return"float";case E.Int32:return"int";case E.Uint32:return"uint";case E.Bool:return"bool";default:throw new Error(`Unsupported dtype for WebGL: ${e}`)}}function ne(e){const r=`load_${e}`,o=D(e);let t;if(L(e))t="val";else if(e===E.Int32)t="floatBitsToInt(val)";else if(e===E.Uint32)t="floatBitsToUint(val)";else if(e===E.Bool)t="floatBitsToInt(val) != 0";else throw new Error(`Unsupported dtype for WebGL fetch: ${e}`);return`
${o} ${r}(highp sampler2D tex, int idx) {
  ivec2 texSize = textureSize(tex, 0);
  int texel = idx / 4;
  int component = idx - texel * 4;
  ivec2 coord = ivec2(texel % texSize.x, texel / texSize.x);
  vec4 texVal = texelFetch(tex, coord, 0);
  float val;
  if (component == 0) val = texVal.x;
  else if (component == 1) val = texVal.y;
  else if (component == 2) val = texVal.z;
  else val = texVal.w;
  return ${t};
}
`}function ie(e,r){switch(e){case E.Float32:return r;case E.Int32:return`intBitsToFloat(${r})`;case E.Uint32:return`uintBitsToFloat(${r})`;case E.Bool:return`intBitsToFloat(${r} ? 1 : 0)`;default:throw new Error(`Unsupported dtype for WebGL output: ${e}`)}}function N(e,r){switch(e){case E.Bool:return r?"true":"false";case E.Int32:return r.toString();case E.Uint32:return r.toString()+"u";case E.Float32:return Number.isNaN(r)?"uintBitsToFloat(0x7fc00000u)":Number.isFinite(r)?"float("+r.toString()+")":r>0?"uintBitsToFloat(0x7f800000u)":"uintBitsToFloat(0xff800000u)";default:throw new Error(`Unsupported dtype for WebGL constant: ${e}`)}}function O(e,r,o){const t=new Map,n=_=>{if(t.has(_))return t.get(_);const{op:a,src:x,dtype:u,arg:l}=_;let i="";if(P.Binary.has(a)){const s=n(x[0]),f=n(x[1]);a===c.Add?u===E.Bool?i=`(${s} || ${f})`:i=`(${s} + ${f})`:a===c.Sub?i=`(${s} - ${f})`:a===c.Mul?u===E.Bool?i=`(${s} && ${f})`:i=`(${s} * ${f})`:a===c.Idiv?L(u)?i=`trunc(${s} / ${f})`:i=`(${s} / ${f})`:a===c.Mod?L(u)?i=`(${s} - ${f} * trunc(${s} / ${f}))`:i=`(${s} % ${f})`:a===c.Min?u===E.Bool?i=`(${s} && ${f})`:i=`min(${s}, ${f})`:a===c.Max&&(u===E.Bool?i=`(${s} || ${f})`:i=`max(${s}, ${f})`)}else if(P.Compare.has(a)){const s=n(x[0]),f=n(x[1]);a===c.Cmplt?i=`(${s} < ${f})`:a===c.Cmpne&&(L(x[0].dtype)?i=`(${s} != ${f} || isnan(${s}) || isnan(${f}))`:i=`(${s} != ${f})`)}else if(P.Unary.has(a)){const s=n(x[0]);if(a===c.Sin)i=`sin(${d(s)})`;else if(a===c.Cos)i=`cos(${d(s)})`;else if(a===c.Asin)i=`asin(${d(s)})`;else if(a===c.Atan)i=`atan(${d(s)})`;else if(a===c.Exp)i=`exp(${d(s)})`;else if(a===c.Log)i=`log(${d(s)})`;else if(a===c.Erf)i=`erf(${d(s)})`;else if(a===c.Erfc)i=`erfc(${d(s)})`;else if(a===c.Sqrt)i=`sqrt(${d(s)})`;else if(a===c.Floor)i=`floor(${d(s)})`;else if(a===c.Ceil)i=`ceil(${d(s)})`;else if(a===c.Reciprocal)i=`(1.0 / ${s})`;else if(a===c.Cast)i=`${D(u)}(${d(s)})`;else if(a===c.Bitcast){const f=x[0].dtype;u===f?i=s:u===E.Float32?f===E.Int32?i=`intBitsToFloat(${d(s)})`:f===E.Uint32&&(i=`uintBitsToFloat(${d(s)})`):u===E.Int32?f===E.Float32?i=`floatBitsToInt(${d(s)})`:f===E.Uint32&&(i=`int(${d(s)})`):u===E.Uint32&&(f===E.Float32?i=`floatBitsToUint(${d(s)})`:f===E.Int32&&(i=`uint(${d(s)})`))}}else if(a===c.Threefry2x32){const[s,f,T,h]=x.map(S=>d(n(S))),R=l,m=`threefry2x32(uvec2(${s}, ${f}), uvec2(${T}, ${h}))`;R==="xor"?i=`(${m}.x ^ ${m}.y)`:R===0?i=`${m}.x`:R===1&&(i=`${m}.y`)}else if(a===c.Where){const[s,f,T]=x.map(n);i=`(${s} ? ${f} : ${T})`}else if(a===c.Const)i=N(u,l);else if(a===c.Special)i=l[0];else if(a===c.Variable)i=l;else if(a===c.GlobalIndex){const s=l[0],f=n(x[0]);i=`load_${o[s]}(${r[s]}, ${d(f)})`}if(!i)throw new J(a,u,"webgl",l);return t.set(_,i),i};return n(e)}export{ce as WebGLBackend};
