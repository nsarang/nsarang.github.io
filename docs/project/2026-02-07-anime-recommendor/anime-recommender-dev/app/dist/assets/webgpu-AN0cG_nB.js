var Ue=Object.defineProperty;var ge=t=>{throw TypeError(t)};var Ce=(t,e,r)=>e in t?Ue(t,e,{enumerable:!0,configurable:!0,writable:!0,value:r}):t[e]=r;var k=(t,e,r)=>Ce(t,typeof e!="symbol"?e+"":e,r),le=(t,e,r)=>e.has(t)||ge("Cannot "+r);var N=(t,e,r)=>(le(t,e,"read from private field"),r?r.call(t):e.get(t)),D=(t,e,r)=>e.has(t)?ge("Cannot add the same private member more than once"):e instanceof WeakSet?e.add(t):e.set(t,r),me=(t,e,r,s)=>(le(t,e,"write to private field"),s?s.call(t,r):e.set(t,r),r),P=(t,e,r)=>(le(t,e,"access private method"),r);import{S as ce,F as Me,E as oe,r as Pe,t as Ee,D as g,m as Ae,A as d,f as re,s as b,a as H,U as Ge,R as J,p as K,b as fe,i as ne,c as _e}from"./index.js";import"./vendor.js";const Le=`
fn threefry2x32(key: vec2<u32>, ctr: vec2<u32>) -> vec2<u32> {
  let ks0: u32 = key.x;
  let ks1: u32 = key.y;
  let ks2: u32 = ks0 ^ ks1 ^ 0x1BD11BDAu;

  var x0: u32 = ctr.x + ks0;
  var x1: u32 = ctr.y + ks1;

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

  return vec2<u32>(x0, x1);
}`,ze=`
const _erf_p: f32 = 0.3275911;
const _erf_a1: f32 = 0.254829592;
const _erf_a2: f32 = -0.284496736;
const _erf_a3: f32 = 1.421413741;
const _erf_a4: f32 = -1.453152027;
const _erf_a5: f32 = 1.061405429;
fn erf(x: f32) -> f32 {
  let t = 1.0 / (1.0 + _erf_p * abs(x));
  let P_t = fma(fma(fma(fma(_erf_a5, t, _erf_a4), t, _erf_a3), t, _erf_a2), t, _erf_a1) * t;
  return sign(x) * (1.0 - P_t * exp(-x * x));
}
fn erfc(x: f32) -> f32 {
  let t = 1.0 / (1.0 + _erf_p * abs(x));
  let P_t = fma(fma(fma(fma(_erf_a5, t, _erf_a4), t, _erf_a3), t, _erf_a2), t, _erf_a1) * t;
  let E = P_t * exp(-x * x);
  return select(2.0 - E, E, x >= 0.0);
}`,ie=String.raw`
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn inf() -> f32 { let bits = 0x7f800000u; return bitcast<f32>(bits); }
`.trim();function E(t,e=!1){switch(t){case g.Bool:return e?"i32":"bool";case g.Int32:return"i32";case g.Uint32:return"u32";case g.Float32:return"f32";case g.Float16:return"f16";default:throw new Error(`Unsupported dtype for WebGPU: ${t}`)}}function Re(t){switch(t){case g.Bool:return"1";case g.Int32:return"2147483647";case g.Uint32:return"4294967295u";case g.Float32:return"inf()";case g.Float16:return"f16(inf())";default:throw new Error(`Unsupported dtype for WebGPU: ${t}`)}}function be(t,e){if(t===g.Bool)return e?"true":"false";if(t===g.Int32)return e.toString();if(t===g.Uint32)return e.toString()+"u";if(t===g.Float32)return Number.isNaN(e)?"nan()":Number.isFinite(e)?"f32("+e.toString()+")":e>0?"inf()":"-inf()";if(t===g.Float16)return Number.isNaN(e)?"f16(nan())":Number.isFinite(e)?"f16("+e.toString()+")":e>0?"f16(inf())":"f16(-inf())";throw new Error(`Unsupported const dtype: ${t}`)}const Y=16384;function se(t){let e=t,r=1;return t>65535&&(e=Y,r=Math.ceil(t/Y)),[e,r]}var j,ue,je,Oe=(j=class{constructor(e){D(this,ue);k(this,"initialized",!1);k(this,"deviceStorage");k(this,"deviceContexts");k(this,"hostStorage");k(this,"hostContext");this.device=e}read(e,r,s){this.initialized||P(this,ue,je).call(this);const i=this.deviceStorage,a=this.deviceContexts,u=this.hostContext,h=Math.ceil(s/4),_=j.width*4,v=new ArrayBuffer(h*4);for(let l=0;l<a.length;l++){const m=a[l].getCurrentTexture(),y=(A,I,ae)=>{const q=this.device.createCommandEncoder();q.copyBufferToTexture({buffer:e,bytesPerRow:_,offset:ae+r},{texture:m},{width:A,height:I,depthOrArrayLayers:1});const V=q.finish();this.device.queue.submit([V]),u.clearRect(0,0,A,I),u.drawImage(i[l],0,0);const O=u.getImageData(0,0,A,I).data,R=new Uint8ClampedArray(v,ae,4*A*I),M=j.alphaModes[l];for(let G=0;G<R.length;G+=4)M==="premultiplied"?R[G+3]=O[G+3]:(R[G]=O[G+2],R[G+1]=O[G+1],R[G+2]=O[G])},$=j.width*j.height,z=Math.floor(h/$);let F=h%$;const T=Math.floor(F/j.width);F=F%j.width;let C=0;for(let A=0;A<z;A++)y(j.width,j.height,C),C+=$*4;T>0&&(y(j.width,T,C),C+=T*j.width*4),F>0&&y(F,1,C)}return new Uint8Array(v,0,s)}},ue=new WeakSet,je=function(){const e=()=>new OffscreenCanvas(j.width,j.height);this.deviceStorage=j.alphaModes.map(e),this.deviceContexts=this.deviceStorage.map((r,s)=>{const i=r.getContext("webgpu");return i.configure({device:this.device,format:"bgra8unorm",usage:GPUTextureUsage.COPY_DST,alphaMode:j.alphaModes[s]}),i}),this.hostStorage=e(),this.hostContext=this.hostStorage.getContext("2d",{willReadFrequently:!0}),this.initialized=!0},k(j,"alphaModes",["opaque","premultiplied"]),k(j,"width",256),k(j,"height",256),j);function Fe(t){const e=new Uint32Array(3);return e[0]=t.kind==="sort"?0:1,e[1]=t.mergeStep??0,e[2]=t.mergeStage??0,new Uint8Array(e.buffer)}function Se(t,e,r,s,i){const a=E(e,!0),u=1<<Math.ceil(Math.log2(r||1)),h=Math.ceil(u/2),_=re(h,t.limits.maxComputeWorkgroupSizeX),v=h/_,l=Math.log2(u),m=Math.min(l,Math.log2(_*2)),y=e===g.Float16,$=ne(e)?`${a}(nan())`:Re(e),z=`
${y?"enable f16;":""}
${ie}

struct Uniforms {
  kind: u32, // 0 = sort, 1 = merge
  merge_step: u32, // half_block = 2^step
  merge_stage: u32, // only used for merge
}

@group(0) @binding(0) var<storage, read> input: array<${a}>;
@group(0) @binding(1) var<storage, read_write> output: array<${a}>;
${i?"@group(0) @binding(2) var<storage, read_write> output_idx: array<i32>;":""}

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

var<workgroup> shared_vals: array<${a}, ${_*2}>;
${i?`var<workgroup> shared_idx: array<i32, ${_*2}>;`:""}

fn compare(a: ${a}, b: ${a}) -> bool {
${ne(e)?`
  let min_value = min(a, b);
  return a == min_value && b != min_value;`:"  return a < b;"}
}

fn compare_and_swap(i: u32, j: u32) {
  let val_i = shared_vals[i];
  let val_j = shared_vals[j];
${i?`
  if (
    compare(val_j, val_i) ||
    (!compare(val_i, val_j) && shared_idx[j] < shared_idx[i])
  ) {
    shared_vals[i] = val_j;
    shared_vals[j] = val_i;
    let tmp_idx = shared_idx[i];
    shared_idx[i] = shared_idx[j];
    shared_idx[j] = tmp_idx;
  }`:`
  if (compare(val_j, val_i)) {
    shared_vals[i] = val_j;
    shared_vals[j] = val_i;
  }`}
}

@compute @workgroup_size(${_})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let blockid = wg_id.x + wg_id.y * ${Y}u;
  let batch = blockid / ${v}u;
  let wg_in_batch = blockid % ${v}u;

  let tid = local_id.x;
  let base = batch * ${r}u;

  if (uniforms.kind == 0u || (uniforms.kind == 1u && uniforms.merge_step == ${m-1}u)) {
    let wg_base = wg_in_batch * ${_*2}u;

    // Load data into shared memory (2 elements per thread)
    let idx0 = tid * 2u;
    let idx1 = tid * 2u + 1u;
    // Load from input for initial 'sort' pass, then from output (read-write) for 'merge' passes.
    if (uniforms.kind == 0u) {
      shared_vals[idx0] = select(${$}, input[base + wg_base + idx0], wg_base + idx0 < ${r}u);
      shared_vals[idx1] = select(${$}, input[base + wg_base + idx1], wg_base + idx1 < ${r}u);
${i?`
      shared_idx[idx0] = i32(wg_base + idx0);
      shared_idx[idx1] = i32(wg_base + idx1);`:""}
    } else {
      shared_vals[idx0] = select(${$}, output[base + wg_base + idx0], wg_base + idx0 < ${r}u);
      shared_vals[idx1] = select(${$}, output[base + wg_base + idx1], wg_base + idx1 < ${r}u);
${i?`
      shared_idx[idx0] = select(${r}, output_idx[base + wg_base + idx0], wg_base + idx0 < ${r}u);
      shared_idx[idx1] = select(${r}, output_idx[base + wg_base + idx1], wg_base + idx1 < ${r}u);`:""}
    }
    workgroupBarrier();

    let initial_stage = select(0u, ${m-1}u, uniforms.kind != 0u);
    for (var stage = initial_stage; stage < ${m}u; stage++) {
      for (var step1 = stage + 1u; step1 > 0u; step1--) {
        let step = step1 - 1u;
        let half_block = 1u << step;
        let is_first_step = uniforms.kind == 0u && step == stage;

        let block_offset = (tid / half_block) * half_block;
        let local_offset = tid % half_block;
        let i = block_offset * 2u + local_offset;
        let j = select(i + half_block, i ^ (half_block * 2u - 1u), is_first_step);
        compare_and_swap(i, j);

        workgroupBarrier();
      }
    }

    if (wg_base + idx0 < ${r}u) {
      output[base + wg_base + idx0] = shared_vals[idx0];
      ${i?"output_idx[base + wg_base + idx0] = shared_idx[idx0];":""}
    }
    if (wg_base + idx1 < ${r}u) {
      output[base + wg_base + idx1] = shared_vals[idx1];
      ${i?"output_idx[base + wg_base + idx1] = shared_idx[idx1];":""}
    }
  } else {
    // Execute single merge pass for a step >= numLocalStages.
    let half_block = 1u << uniforms.merge_step;  // half_block >= workgroupSize * 2
    let thread_in_batch = wg_in_batch * ${_} + tid;
    let is_first_step = uniforms.merge_step == uniforms.merge_stage;

    let block_offset = (thread_in_batch / half_block) * half_block;
    let local_offset = thread_in_batch % half_block;
    let i = block_offset * 2u + local_offset;
    let j = select(i + half_block, i ^ (half_block * 2u - 1u), is_first_step);

    // Global version of compare_and_swap()
    if (j < ${r}u) {
      let val_i = output[base + i];
      let val_j = output[base + j];
${i?`
      let idx_i = output_idx[base + i];
      let idx_j = output_idx[base + j];
      if (compare(val_j, val_i) || (!compare(val_i, val_j) && idx_j < idx_i)) {
        output[base + i] = val_j;
        output[base + j] = val_i;
        output_idx[base + i] = idx_j;
        output_idx[base + j] = idx_i;`:`
      if (compare(val_j, val_i)) {
        output[base + i] = val_j;
        output[base + j] = val_i;`}
      }
    }
  }
}
`.trim(),F=se(s*v),T=[{kind:"sort"}];for(let C=m;C<l;C++)for(let A=C;A>=m-1;A--)T.push({kind:"merge",mergeStep:A,mergeStage:C});return[{code:z,numInputs:1,numOutputs:i?2:1,hasUniform:!0,passes:T.map(C=>({grid:F,uniform:Fe(C)}))}]}function Te(t,e){const r=e.inputDtypes[0],s=e.inputShapes[0],i=s[s.length-1],a=K(s.slice(0,-1));return Se(t,r,i,a,!1)}function Ie(t,e){const r=e.inputDtypes[0],s=e.inputShapes[0],i=s[s.length-1],a=K(s.slice(0,-1));return Se(t,r,i,a,!0)}function We(t,e,r){const s=e.inputDtypes[0],i=e.inputShapes[0],a=e.inputShapes[1],u=i[i.length-1],h=a[a.length-2],_=K(i.slice(0,-2)),v=s===g.Float16,l=E(s,!0),m=re(u,t.limits.maxComputeWorkgroupSizeX),y=`
${v?"enable f16;":""}
${ie}

@group(0) @binding(0) var<storage, read> a: array<${l}>;
@group(0) @binding(1) var<storage, read> b: array<${l}>;
@group(0) @binding(2) var<storage, read_write> x: array<${l}>;

// Shared memory for the current pivot value x[j]
var<workgroup> x_j: ${l};

@compute @workgroup_size(${m})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let wg_idx = wg_id.x + wg_id.y * ${Y}u;
  let mat_idx = wg_idx / ${h}u;
  let rhs_idx = wg_idx % ${h}u;

  if (mat_idx >= ${_}u) {
    return;
  }

  let a_base = mat_idx * ${u*u}u;
  let bx_base = (mat_idx * ${h}u + rhs_idx) * ${u}u;
  let tid = local_id.x;

  // Step 1: Copy b to x (threads collaborate)
  for (var idx = tid; idx < ${u}u; idx += ${m}u) {
    x[bx_base + idx] = b[bx_base + idx];
  }
  storageBarrier();

  // Step 2: Back-substitution from j = n-1 down to 0
  for (var jj = 0u; jj < ${u}u; jj++) {
    let j = ${u-1}u - jj;

    // Thread 0 computes x[j] = x[j] / a[j,j]
    if (tid == 0u) {
      ${r.unitDiagonal?"x_j = x[bx_base + j];":`x_j = x[bx_base + j] / a[a_base + j * ${u}u + j];`}
      x[bx_base + j] = x_j;
    }
    workgroupBarrier();  // Sync shared memory x_j

    // All threads subtract x[j] * a[i,j] from x[i] for i < j
    for (var i = tid; i < j; i += ${m}u) {
      x[bx_base + i] -= x_j * a[a_base + i * ${u}u + j];
    }
    workgroupBarrier();
    storageBarrier();
  }
}
`.trim(),$=_*h,z=se($);return[{code:y,numInputs:2,numOutputs:1,hasUniform:!1,passes:[{grid:z}]}]}function qe(t,e){const r=e.inputDtypes[0],s=e.inputShapes[0],i=s[s.length-1],a=K(s.slice(0,-2)),u=r===g.Float16,h=E(r,!0),_=re(i,t.limits.maxComputeWorkgroupSizeX),v=`
${u?"enable f16;":""}
${ie}

@group(0) @binding(0) var<storage, read> input: array<${h}>;
@group(0) @binding(1) var<storage, read_write> output: array<${h}>;

// Shared memory for the diagonal element
var<workgroup> L_jj: ${h};

@compute @workgroup_size(${_})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let batch = wg_id.x + wg_id.y * ${Y}u;
  if (batch >= ${a}u) {
    return;
  }

  let base = batch * ${i*i}u;
  let tid = local_id.x;

  // Zero out output and copy lower triangle from input (threads collaborate)
  for (var idx = tid; idx < ${i*i}u; idx += ${_}u) {
    let row = idx / ${i}u;
    let col = idx % ${i}u;
    output[base + idx] = select(0, input[base + idx], col <= row);
  }
  storageBarrier();

  // Cholesky-Crout algorithm: process column by column
  for (var j = 0u; j < ${i}u; j++) {
    // Step 1: All threads compute sum for their rows i >= j in parallel
    // sum = A[i][j] - sum(L[i][k] * L[j][k] for k < j)
    for (var i = j + tid; i < ${i}u; i += ${_}u) {
      var sum = output[base + i * ${i}u + j];
      for (var k = 0u; k < j; k++) {
        sum -= output[base + i * ${i}u + k] * output[base + j * ${i}u + k];
      }
      output[base + i * ${i}u + j] = sum;
    }
    storageBarrier();

    // Step 2: Thread 0 computes L[j][j] = sqrt(output[j][j])
    if (tid == 0u) {
      L_jj = sqrt(output[base + j * ${i}u + j]);
      output[base + j * ${i}u + j] = L_jj;
    }
    workgroupBarrier();

    // Step 3: All threads divide output[i][j] by L[j][j] for i > j
    for (var i = j + 1u + tid; i < ${i}u; i += ${_}u) {
      output[base + i * ${i}u + j] /= L_jj;
    }
    storageBarrier();
  }
}
`.trim(),l=se(a);return[{code:v,numInputs:1,numOutputs:1,hasUniform:!1,passes:[{grid:l}]}]}function Ne(t,e){const r=e.inputDtypes[0],s=e.inputShapes[0],i=s[s.length-2],a=s[s.length-1],u=Math.min(i,a),h=K(s.slice(0,-2)),_=r===g.Float16,v=E(r,!0),l=re(Math.max(i,a),t.limits.maxComputeWorkgroupSizeX),m=`
${_?"enable f16;":""}
${ie}

@group(0) @binding(0) var<storage, read> input: array<${v}>;
@group(0) @binding(1) var<storage, read_write> lu: array<${v}>;
@group(0) @binding(2) var<storage, read_write> pivots: array<i32>;
@group(0) @binding(3) var<storage, read_write> perm: array<i32>;

var<workgroup> pivot_row: u32;
var<workgroup> pivot_val: ${v};

@compute @workgroup_size(${l})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let batch = wg_id.x + wg_id.y * ${Y}u;
  if (batch >= ${h}u) {
    return;
  }

  let lu_base = batch * ${i*a}u;
  let piv_base = batch * ${u}u;
  let perm_base = batch * ${i}u;
  let tid = local_id.x;

  // Copy input to lu
  for (var idx = tid; idx < ${i*a}u; idx += ${l}u) {
    lu[lu_base + idx] = input[lu_base + idx];
  }
  // Initialize permutation
  for (var idx = tid; idx < ${i}u; idx += ${l}u) {
    perm[perm_base + idx] = i32(idx);
  }
  storageBarrier();

  // LU decomposition with partial pivoting
  for (var j = 0u; j < ${u}u; j++) {
    // Step 1: Thread 0 finds pivot (max abs value in column j, rows >= j)
    if (tid == 0u) {
      var max_val = abs(lu[lu_base + j * ${a}u + j]);
      var max_row = j;
      for (var i = j + 1u; i < ${i}u; i++) {
        let val = abs(lu[lu_base + i * ${a}u + j]);
        if (val > max_val) {
          max_val = val;
          max_row = i;
        }
      }
      pivot_row = max_row;
      pivot_val = lu[lu_base + max_row * ${a}u + j];
      pivots[piv_base + j] = i32(max_row);
    }
    workgroupBarrier();

    // Step 2: Swap rows j and pivot_row (threads collaborate)
    let pr = pivot_row;
    if (pr != j) {
      for (var col = tid; col < ${a}u; col += ${l}u) {
        let tmp = lu[lu_base + j * ${a}u + col];
        lu[lu_base + j * ${a}u + col] = lu[lu_base + pr * ${a}u + col];
        lu[lu_base + pr * ${a}u + col] = tmp;
      }
      if (tid == 0u) {
        let tmp_p = perm[perm_base + j];
        perm[perm_base + j] = perm[perm_base + pr];
        perm[perm_base + pr] = tmp_p;
      }
    }
    storageBarrier();

    // Step 3: Compute L[i][j] and update submatrix
    // Each thread handles one row i > j
    for (var i = j + 1u + tid; i < ${i}u; i += ${l}u) {
      let factor = lu[lu_base + i * ${a}u + j] / pivot_val;
      lu[lu_base + i * ${a}u + j] = factor; // L[i][j]
      for (var k = j + 1u; k < ${a}u; k++) {
        lu[lu_base + i * ${a}u + k] -= factor * lu[lu_base + j * ${a}u + k];
      }
    }
    storageBarrier();
  }
}
`.trim(),y=se(h);return[{code:m,numInputs:1,numOutputs:3,hasUniform:!1,passes:[{grid:y}]}]}function $e(t,e){switch(e.name){case J.Sort:return Te(t,e.type);case J.Argsort:return Ie(t,e.type);case J.TriangularSolve:return We(t,e.type,e.params);case J.Cholesky:return qe(t,e.type);case J.LU:return Ne(t,e.type);default:throw new Ge(e.name,"webgpu")}}var ee,W,U,pe,Q,X,ve,Je=(ve=class{constructor(t){D(this,U);k(this,"type","webgpu");k(this,"maxArgs");k(this,"pipelines");k(this,"syncReader");k(this,"buffers");k(this,"nextSlot");D(this,ee,new Map);D(this,W);this.device=t,this.maxArgs=this.device.limits.maxStorageBuffersPerShaderStage-1,this.pipelines=new Ke(t),this.syncReader=new Oe(t),this.buffers=new Map,this.nextSlot=1,me(this,W,P(this,U,X).call(this,4)),t.addEventListener("uncapturederror",e=>{console.error("Uncaptured error in WebGPU backend:",e.error.message)})}malloc(t,e){let r;const s=Math.ceil(t/4)*4;if(t===0)r=N(this,W);else if(e){if(e.byteLength!==t)throw new Error("initialData size does not match buffer size");if(e.byteLength<4096)r=P(this,U,X).call(this,s,{mapped:!0}),new Uint8Array(r.getMappedRange(),0,t).set(e),r.unmap();else if(r=P(this,U,X).call(this,s),e.byteLength%4===0)this.device.queue.writeBuffer(r,0,e);else{const a=e.byteLength-e.byteLength%4;this.device.queue.writeBuffer(r,0,e,0,a);const u=new Uint8Array(4);u.set(e.subarray(a)),this.device.queue.writeBuffer(r,a,u)}}else r=P(this,U,X).call(this,s);const i=this.nextSlot++;return this.buffers.set(i,{buffer:r,size:t,ref:1}),i}incRef(t){const e=this.buffers.get(t);if(!e)throw new ce(t);e.ref++}decRef(t){const e=this.buffers.get(t);if(!e)throw new ce(t);e.ref--,e.ref===0&&(this.buffers.delete(t),e.buffer!==N(this,W)&&e.buffer.destroy())}async read(t,e,r){const{buffer:s,size:i}=P(this,U,Q).call(this,t);if(s===N(this,W))return new Uint8Array;e===void 0&&(e=0),r===void 0&&(r=i-e);const a=Math.ceil(r/4)*4,u=P(this,U,X).call(this,a,{read:!0});try{const h=this.device.createCommandEncoder();h.copyBufferToBuffer(s,e,u,0,a),this.device.queue.submit([h.finish()]),await u.mapAsync(GPUMapMode.READ);const _=u.getMappedRange();return new Uint8Array(_.slice(),0,r)}finally{u.destroy()}}readSync(t,e,r){const{buffer:s,size:i}=P(this,U,Q).call(this,t);return s===N(this,W)?new Uint8Array:(e===void 0&&(e=0),r===void 0&&(r=i-e),this.syncReader.read(s,e,r))}async prepareKernel(t){const e=P(this,U,pe).call(this,t),r=await this.pipelines.prepare(e);return new oe(t,[{...e,pipeline:r}])}prepareKernelSync(t){const e=P(this,U,pe).call(this,t),r=this.pipelines.prepareSync(e);return new oe(t,[{...e,pipeline:r}])}async prepareRoutine(t){const e=$e(this.device,t),r=await Promise.all(e.map(async s=>{const i=await this.pipelines.prepare(s);return{...s,pipeline:i}}));return new oe(t,r)}prepareRoutineSync(t){const r=$e(this.device,t).map(s=>{const i=this.pipelines.prepareSync(s);return{...s,pipeline:i}});return new oe(t,r)}dispatch(t,e,r){const s=e.map(a=>P(this,U,Q).call(this,a).buffer),i=r.map(a=>P(this,U,Q).call(this,a).buffer);Xe(this.device,t.data,s,i)}},ee=new WeakMap,W=new WeakMap,U=new WeakSet,pe=function(t){const e=Me.hash(t);let r=N(this,ee).get(e);return r||(r=De(this.device,t),N(this,ee).set(e,r)),r},Q=function(t){const e=this.buffers.get(t);if(!e)throw new ce(t);return{buffer:e.buffer,size:e.size}},X=function(t,{mapped:e=!1,read:r=!1}={}){if(r&&e)throw new Error("mapped and read cannot both be true");return this.device.createBuffer({size:t,usage:r?GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,mappedAtCreation:e})},ve);function De(t,e){var G,xe,he;const r=Ee(e),{nargs:s,reduction:i}=e,a=Array.from({length:s},(p,n)=>`in${n}`),u=[];let h="";const _=Symbol("pushIndent"),v=Symbol("popIndent"),l=(...p)=>{for(const n of p)n===_?h+="  ":n===v?h=h.slice(0,-2):u.push(n&&h+n)};if(r.exp.some(p=>p.dtype===g.Float16)||(G=r.epilogue)!=null&&G.some(p=>p.dtype===g.Float16)){if(!t.features.has("shader-f16"))throw new Error("WebGPU device does not support shader-f16 feature");l("enable f16;")}l(ie);const m=Ae(r.exp.distinctOps(),(xe=r.epilogue)==null?void 0:xe.distinctOps());m.has(d.Threefry2x32)&&l(Le),(m.has(d.Erf)||m.has(d.Erfc))&&l(ze),l("");const y=Array.from({length:s},()=>null);r.exp.fold(p=>{p.op===d.GlobalIndex&&(y[p.arg[0]]=p.dtype)}),(he=r.epilogue)==null||he.fold(p=>{p.op===d.GlobalIndex&&(y[p.arg[0]]=p.dtype)});for(let p=0;p<s;p++){const n=E(y[p]??g.Float32,!0);l(`@group(0) @binding(${p}) var<storage, read> ${a[p]} : array<${n}>;`)}const $=E(e.dtype,!0);l(`@group(0) @binding(${s}) var<storage, read_write> result : array<${$}>;`);const z=re(r.threadCount,256),F=Math.ceil(r.threadCount/z),[T,C]=se(F);if(l("",`@compute @workgroup_size(${z})`,"fn main(@builtin(global_invocation_id) id : vec3<u32>) {",_),C===1)l(`if (id.x >= ${r.threadCount}) { return; }`,"let gidx: i32 = i32(id.x);");else{const p=T*z;l(`if (${p} * id.y + id.x >= ${r.threadCount}) { return; }`,`let gidx: i32 = i32(${p} * id.y + id.x);`)}let A=0;const I=()=>`alu${A++}`,ae=p=>p.match(/^alu[0-9]+$/);a.length>0&&l(a.map(p=>`_ = &${p};`).join(" "));const q=new Map,V=new Set,O=p=>{if(q.set(p,(q.get(p)??0)+1),!V.has(p)){V.add(p);for(const n of p.src)O(n)}},R=new Map,M=p=>{if(R.has(p))return R.get(p);const{op:n,src:w,dtype:S,arg:L}=p;let f="";if(fe.Binary.has(n)||fe.Compare.has(n)){const c=M(w[0]),o=M(w[1]);if(n===d.Add)S===g.Bool?f=`(${c} || ${o})`:f=`(${c} + ${o})`;else if(n===d.Sub)f=`(${c} - ${o})`;else if(n===d.Mul)S===g.Bool?f=`(${c} && ${o})`:f=`(${c} * ${o})`;else if(n===d.Idiv)f=ne(S)?`trunc(${c} / ${o})`:`(${c} / ${o})`;else if(n===d.Mod)f=`(${c} % ${o})`;else if(n===d.Min)S===g.Bool?f=`(${c} && ${o})`:f=`min(${b(c)}, ${b(o)})`;else if(n===d.Max)S===g.Bool?f=`(${c} || ${o})`:f=`max(${b(c)}, ${b(o)})`;else if(n===d.Cmplt)f=`(${c} < ${o})`;else if(n===d.Cmpne)if(ne(w[0].dtype)){const x=ae(c)?c:I();x!==c&&l(`let ${x} = ${c};`),f=`(${x} != ${o} || min(${x}, ${E(w[0].dtype)}(inf())) != ${x})`}else f=`(${c} != ${o})`}else if(fe.Unary.has(n))if(n===d.Reciprocal&&w[0].op===d.Sqrt)f=`inverseSqrt(${M(w[0].src[0])})`;else{const c=M(w[0]);if(n===d.Sin)f=`sin(${b(c)})`;else if(n===d.Cos)f=`cos(${b(c)})`;else if(n===d.Asin)f=`asin(${b(c)})`;else if(n===d.Atan)f=`atan(${b(c)})`;else if(n===d.Exp)f=`exp(${b(c)})`;else if(n===d.Log)f=`log(${b(c)})`;else if(n===d.Erf||n===d.Erfc){const o=n===d.Erf?"erf":"erfc";S!==g.Float32?f=`${E(S)}(${o}(f32(${b(c)})))`:f=`${o}(${b(c)})`}else n===d.Sqrt?f=`sqrt(${b(c)})`:n===d.Reciprocal?f=`(1.0 / ${c})`:n===d.Floor?f=`floor(${b(c)})`:n===d.Ceil?f=`ceil(${b(c)})`:n===d.Cast?f=`${E(S)}(${b(c)})`:n===d.Bitcast&&(f=`bitcast<${E(S)}>(${b(c)})`)}else if(n===d.Where)f=`select(${b(M(w[2]))}, ${b(M(w[1]))}, ${b(M(w[0]))})`;else if(n===d.Threefry2x32){const c=I(),[o,x,B,ke]=w.map(Be=>b(M(Be)));if(l(`let ${c} = threefry2x32(vec2(${o}, ${x}), vec2(${B}, ${ke}));`),L==="xor")f=`(${c}.x ^ ${c}.y)`;else if(L===0)f=`${c}.x`;else if(L===1)f=`${c}.y`;else throw new _e(n,S,"webgpu",L)}else{if(n===d.Const)return be(S,L);if(n===d.Special)return L[0];if(n===d.Variable)return L;n===d.GlobalIndex&&(f=`${a[L[0]]}[${b(M(w[0]))}]`,S===g.Bool&&(f=`(${f} != 0)`))}if(!f)throw new _e(n,S,"webgpu",L);const Z=E(S);if((q.get(p)??0)>1){const c=I();return R.set(p,c),l(`let ${c}: ${Z} = ${b(f)};`),c}else return R.set(p,f),f};if(i){if((r.size.groups??1)>1)throw new Error("WebGPU backend does not support group optimization yet");const p=r.size.unroll??1,n=r.size.upcast??1,w=[...Array(n)].map((o,x)=>`acc${x}`);for(let o=0;o<n;o++)l(`var ${w[o]}: ${E(i.dtype)} = ${be(i.dtype,i.identity)};`);l(`for (var ridx: i32 = 0; ridx < ${r.size.reduce}; ridx++) {`,_);const S=[],L=new Map;for(let o=0;o<n;o++){S.push([]);for(let x=0;x<p;x++){const B=r.exp.substitute({upcast:H.i32(o),unroll:H.i32(x)});S[o].push(B.simplify(L)),O(S[o][x])}}const f=S.map(o=>o.map(M).map(b));for(let o=0;o<n;o++){let x=f[o][0];for(let B=1;B<p;B++)if(i.op===d.Add)x=`${x} + ${f[o][B]}`;else if(i.op===d.Mul)x=`${x} * ${f[o][B]}`;else if(i.op===d.Min)x=i.dtype===g.Bool?`(${x} && ${f[o][B]})`:`min(${x}, ${f[o][B]})`;else if(i.op===d.Max)x=i.dtype===g.Bool?`(${x} || ${f[o][B]})`:`max(${x}, ${f[o][B]})`;else throw new Error(`Unsupported reduction op: ${i.op}`);if(i.op===d.Add)l(`${w[o]} += ${x};`);else if(i.op===d.Mul)l(`${w[o]} *= ${x};`);else if(i.op===d.Min)i.dtype===g.Bool?l(`${w[o]} = ${w[o]} && ${x};`):l(`${w[o]} = min(${w[o]}, ${x});`);else if(i.op===d.Max)i.dtype===g.Bool?l(`${w[o]} = ${w[o]} || ${x};`):l(`${w[o]} = max(${w[o]}, ${x});`);else throw new Error(`Unsupported reduction op: ${i.op}`)}l(v,"}"),R.clear(),q.clear(),V.clear();const Z=[],c=[];for(let o=0;o<n;o++){const x=r.outputIdxExp.substitute({upcast:H.i32(o)});Z.push(x.simplify(L)),O(Z[o]),c.push(r.epilogue.substitute({acc:H.variable(i.dtype,w[o]),upcast:H.i32(o)}).simplify(L)),O(c[o])}for(let o=0;o<n;o++){const x=b(M(Z[o]));let B=b(M(c[o]));$!==E(c[o].dtype)&&(B=`${$}(${B})`),l(`result[${x}] = ${B};`)}}else{O(r.exp);let p=b(M(r.exp));$!==E(r.exp.dtype)&&(p=`${$}(${p})`),l(`result[gidx] = ${p};`)}return l(v,"}"),{code:u.join(`
`),numInputs:s,numOutputs:1,hasUniform:!1,passes:[{grid:[T,C]}]}}function Xe(t,e,r,s){const i=t.createCommandEncoder();for(const{pipeline:a,...u}of e){if(r.length!==u.numInputs||s.length!==u.numOutputs)throw new Error(`webgpu: expected ${u.numInputs} inputs and ${u.numOutputs} outputs, got ${r.length} inputs and ${s.length} outputs`);const h=u.passes.filter(({grid:m})=>K(m)>0);if(h.length===0)continue;const _=t.createBindGroup({layout:a.getBindGroupLayout(0),entries:[...r.map((m,y)=>({binding:y,resource:{buffer:m}})),...s.map((m,y)=>({binding:r.length+y,resource:{buffer:m}}))]});let v=null,l=0;if(u.hasUniform){const m=h.map(({uniform:z})=>z),[y,$]=Ye(t,m);l=$,v=t.createBindGroup({layout:a.getBindGroupLayout(1),entries:[{binding:0,resource:{buffer:y,size:$}}]})}for(let m=0;m<h.length;m++){const{grid:y}=h[m],$=i.beginComputePass();$.setPipeline(a),$.setBindGroup(0,_),v&&$.setBindGroup(1,v,[m*l]),$.dispatchWorkgroups(y[0],y[1]),$.end()}}t.queue.submit([i.finish()])}function Ye(t,e){for(const u of e)if(!u||u.byteLength===0||u.byteLength!==e[0].byteLength)throw new Error("webgpu: Uniform mismatch between shader passes");const r=t.limits.minUniformBufferOffsetAlignment,s=Math.ceil(e[0].byteLength/r)*r,i=t.createBuffer({size:s*e.length,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:!0}),a=new Uint8Array(i.getMappedRange());for(let u=0;u<e.length;u++)a.set(e[u],u*s);return i.unmap(),[i,s]}var te,de,ye,Ke=(ye=class{constructor(t){D(this,te);k(this,"cache");k(this,"inProgress");this.device=t,this.cache=new Map,this.inProgress=new Map}async prepare(t){const e=this.cache.get(t.code);if(e)return e;const r=this.inProgress.get(t.code);if(r)return await r;const s=this.device.createShaderModule({code:t.code}),i=(async()=>{this.device.pushErrorScope("validation");try{const u=await this.device.createComputePipelineAsync({layout:P(this,te,de).call(this,t),compute:{module:s,entryPoint:"main"}});return await this.device.popErrorScope(),u}catch{const h=await this.device.popErrorScope(),_=await we(s,h,t.code);throw new Error(_)}})();this.inProgress.set(t.code,i);const a=await i;return this.cache.set(t.code,a),a}prepareSync(t){const e=this.cache.get(t.code);if(e)return e;const r=this.device.createShaderModule({code:t.code});this.device.pushErrorScope("validation");const s=this.device.createComputePipeline({layout:P(this,te,de).call(this,t),compute:{module:r,entryPoint:"main"}});return this.device.popErrorScope().then(async i=>{if(i!==null){const a=await we(r,i,t.code);console.error(a)}}),this.cache.set(t.code,s),s}},te=new WeakSet,de=function(t){if(t.numInputs+t.numOutputs>this.device.limits.maxStorageBuffersPerShaderStage){const r=t.numInputs+t.numOutputs,s=this.device.limits.maxStorageBuffersPerShaderStage;throw new Error(`Too many buffers (${r}) for WebGPU pipeline (max: ${s})`)}const e=[this.device.createBindGroupLayout({entries:Pe(t.numInputs+t.numOutputs).map(r=>({binding:r,visibility:GPUShaderStage.COMPUTE,buffer:{type:r<t.numInputs?"read-only-storage":"storage"}}))})];return t.hasUniform&&e.push(this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform",hasDynamicOffset:!0}}]})),this.device.createPipelineLayout({bindGroupLayouts:e})},ye);async function we(t,e,r){let s=`Failed to compile shader: ${e?e.message:"(no error scope)"}`;const i=await t.getCompilationInfo();for(const a of i.messages)s+=`
  [${a.type} at ${a.lineNum}:${a.linePos}] ${a.message}`;return r&&(s+=`

${r}`),s}export{Je as WebGPUBackend};
