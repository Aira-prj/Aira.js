/**
 * πirφ.js
 * Project Maded by shadow(Ömer Faruk Açar)
 * Aira is a artificial intelligence(GPT algorithm) library that is currently under development.
 * GPU Version
 */

//require('fs')
//---------------------------------------
//Starting and defining WebGPU
//---------------------------------------
let debug = false
async function initGPU() {
    if (!('gpu' in navigator)) return null
  const adapter = await navigator.gpu.requestAdapter({performance:"high-performance"})
  if (!adapter) return null
  const device = await adapter.requestDevice()
  const queue = device.queue

  device.lost.then(info => {
    console.error("GPU device lost:", info)
  }).catch(()=>{})
  return device
}
function toFloat32ArrayMatrix(mat) {
    if (!Array.isArray(mat) || mat.length === 0 || !Array.isArray(mat[0])) {
    return { arr: new Float32Array(0), rows: 0, cols: 0 }
  }
  const rows = mat.length
  const cols = mat[0].length
  const arr = new Float32Array(rows*cols)
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      arr[i * cols + j] = mat[i][j]
    }
  }
  if(debug) {
    console.log(arr, rows, cols)
  }
  return { arr, rows, cols }
}

function fromFloat32ArrayMatrix(arr,rows,cols) {
  const M=Array.from({length:rows},() => Array(cols))

  for(let i=0;i<rows;i++) {
    for(let j=0;j<cols;j++) {
      M[i][j] = arr[i*cols+j]
    }
  }
  
  if(debug) {
    console.log(M)
  }
  return M
}

async function matMul(device, A, B) {
  if (!device) throw new Error("Device not found")

  const m = A.length
  const k = A[0].length
  const n = B[0].length

  if (B.length !== k)
    throw new Error("Inner dimensions mismatch")

  const aArr = new Float32Array(m * k)
  const bArr = new Float32Array(k * n)

  for (let i = 0; i < m; i++)
    for (let t = 0; t < k; t++)
      aArr[i * k + t] = A[i][t]

  for (let t = 0; t < k; t++)
    for (let j = 0; j < n; j++)
      bArr[t * n + j] = B[t][j]

  const aBuf = device.createBuffer({
    size: aArr.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  })
  device.queue.writeBuffer(aBuf, 0, aArr)

  const bBuf = device.createBuffer({
    size: bArr.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  })
  device.queue.writeBuffer(bBuf, 0, bArr)

  const cBuf = device.createBuffer({
    size: m * n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  })

  const uniform = new Uint32Array([m, k, n])
  const uniformBuf = device.createBuffer({
    size: uniform.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true
  })
  new Uint32Array(uniformBuf.getMappedRange()).set(uniform)
  uniformBuf.unmap()

  const shader = `
struct Matrix { data: array<f32> };

@group(0) @binding(0) var<storage, read> A : Matrix;
@group(0) @binding(1) var<storage, read> B : Matrix;
@group(0) @binding(2) var<storage, read_write> C : Matrix;
@group(0) @binding(3) var<uniform> dims : vec3<u32>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {

    let row = gid.x;
    let col = gid.y;

    let M = dims.x;
    let K = dims.y;
    let N = dims.z;

    if (row >= M || col >= N) { return; }

    var sum : f32 = 0.0;

    for (var t:u32 = 0u; t < K; t = t + 1u) {
        let a = A.data[row * K + t];
        let b = B.data[t * N + col];
        sum = sum + a * b;
    }

    C.data[row * N + col] = sum;
}
`

  const module = device.createShaderModule({ code: shader })
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" }
  })

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: bBuf } },
      { binding: 2, resource: { buffer: cBuf } },
      { binding: 3, resource: { buffer: uniformBuf } },
    ]
  })

  const encoder = device.createCommandEncoder()
  const pass = encoder.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)

  pass.dispatchWorkgroups(
    Math.ceil(m / 8),
    Math.ceil(n / 8)
  )

  pass.end()

  const readBuf = device.createBuffer({
    size: m * n * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  })

  encoder.copyBufferToBuffer(cBuf, 0, readBuf, 0, m * n * 4)
  device.queue.submit([encoder.finish()])

  await readBuf.mapAsync(GPUMapMode.READ)
  const result = new Float32Array(readBuf.getMappedRange().slice(0))
  readBuf.unmap()

  const C = []
  for (let i = 0; i < m; i++) {
    C.push(Array.from(result.slice(i * n, (i + 1) * n)))
  }

  return C
}

async function matVecMul(device,mat,vec) {
  if(!device) console.log("Device is not found")
    
  const queue = device.queue
  const gpu = device
  const { arr: matArr, rows, cols } = toFloat32ArrayMatrix(mat);
  const vecArr = new Float32Array(vec);
  const calculatedRows = Math.max(rows, 1);
  const calculatedOutSize = calculatedRows * 4;
  const matBuf = device.createBuffer({
    size: Math.max(matArr.byteLength, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(matBuf, 0, matArr);

  const vecBuf = device.createBuffer({
    size: Math.max(vecArr.byteLength, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vecBuf, 0, vecArr);
  const outBuf = device.createBuffer({
    size: Math.max(calculatedOutSize, 16), 
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const uniform = new Uint32Array([rows,cols])
  const uniformBuf = gpu.createBuffer({
    size:uniform.byteLength,
    usage:GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation:true
  })
  new Uint32Array(uniformBuf.getMappedRange()).set(uniform)
  uniformBuf.unmap()
  const shaderCode = `
struct Mat { 
  data: array<f32> 
};
struct Vec { 
  data: array<f32> 
};
struct Out { 
  data: array<f32> 
};

@group(0) @binding(1) var<storage, read> mat : Mat;
@group(0) @binding(2) var<storage, read> vecbuf : Vec;
@group(0) @binding(3) var<storage, read_write> outbuf : Out;
@group(0) @binding(4) var<uniform> dims : vec2<u32>;

@compute @workgroup_size(8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let row = gid.x;
    if (row >= dims.x) {
        return;
    };
    var sum : f32 = 0.0;
    let cols = dims.y;
    var j : u32 = 0u;
    loop {
        if (j >= cols) {
            break;
        };
        let a = mat.data[row * cols + j];
        let b = vecbuf.data[j];
        sum = sum + a * b;
        j = j + 1u;
    };
    outbuf.data[row] = sum;
}

  `
  const module = gpu.createShaderModule({code:shaderCode})
  const pipeline = gpu.createComputePipeline({
    layout:'auto',
    compute:{module,entryPoint:'main'}

  })
  
const bindGroup = gpu.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 1, resource: { buffer: matBuf } },
    { binding: 2, resource: { buffer: vecBuf } },
    { binding: 3, resource: { buffer: outBuf } },
    { binding: 4, resource: { buffer: uniformBuf } }
  ]
})

const commandEncoder = gpu.createCommandEncoder()
const pass = commandEncoder.beginComputePass() 
pass.setPipeline(pipeline) 
pass.setBindGroup(0,bindGroup)
const gx = Math.max(Math.ceil(calculatedRows / 64))
const gy = 1
pass.dispatchWorkgroups(gx, gy);
pass.end()
const outSize = Float32Array.BYTES_PER_ELEMENT * rows
const readBuf = gpu.createBuffer({
  size:outSize,
  usage:GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
})

commandEncoder.copyBufferToBuffer(outBuf,0,readBuf,0,outSize)
queue.submit([commandEncoder.finish()])

await readBuf.mapAsync(GPUMapMode.READ)
const copyArrayBuffer = readBuf.getMappedRange()
const result = new Float32Array(copyArrayBuffer.slice(0))
readBuf.unmap()
const out = Array.from(result)
if(debug) {
    console.log(out)
}
return out 
}


async function Outer(device,a,b) {
  if(!device) {
    return console.log("Device is not found")
    
}

const gpu = device
const queue = device.queue
const aArr = new Float32Array(a)
const bArr = new Float32Array(b)
const m = a.length
const n = b.length
const aBuf = gpu.createBuffer({
  size:aArr.byteLength,
  usage:GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  mappedAtCreation:true 

})
new Float32Array(aBuf.getMappedRange()).set(aArr)
aBuf.unmap()
const bBuf = gpu.createBuffer({
  size:bArr.byteLength,
  usage:GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  mappedAtCreation:true

})
new Float32Array(bBuf.getMappedRange()).set(bArr)
bBuf.unmap()
const outSize = Float32Array.BYTES_PER_ELEMENT * m * n
const outBuf = gpu.createBuffer({
  size:outSize,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
})

const uniform = new Uint32Array([m,n])
const uniformBuf = gpu.createBuffer({
  size:uniform.byteLength,
  usage:GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  mappedAtCreation:true
})
new Uint32Array(uniformBuf.getMappedRange()).set(uniform)
uniformBuf.unmap()

const shader = `
struct A { 
  data: array<f32> 
};
struct B { 
  data: array<f32> 
};
struct Out {
  data: array<f32> 
};
@group(0) @binding(1) var<storage, read> a : A;
@group(0) @binding(2) var<storage, read> b : B;
@group(0) @binding(3) var<storage, read_write> outbuf : Out;
@group(0) @binding(4) var<uniform> dims : vec2<u32>;

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  if (i >= dims.x || j >= dims.y) { return; }
  let val = a.data[i] * b.data[j];
  outbuf.data[i * dims.y + j] = val;
}
`
const module = gpu.createShaderModule({code:shader})
const pipeline = gpu.createComputePipeline({
  layout:'auto',
  compute: { module,entryPoint:'main' }
})
const bindGroup = gpu.createBindGroup({
  layout:pipeline.getBindGroupLayout(0),
  entries:[
      { binding: 1, resource: { buffer: aBuf } },
      { binding: 2, resource: { buffer: bBuf } },
      { binding: 3, resource: { buffer: outBuf } },
      { binding: 4, resource: { buffer: uniformBuf } }
  ]
})
const commandEncoder = gpu.createCommandEncoder()
const pass = commandEncoder.beginComputePass()
pass.setPipeline(pipeline)
pass.setBindGroup(0,bindGroup)
const gx = Math.ceil(m/16)
const gy = Math.ceil(n/16)
pass.dispatchWorkgroups(gx,gy)

pass.end()

const outBufSize = Float32Array.BYTES_PER_ELEMENT * m * n
const readBuf = gpu.createBuffer({
  size:outBufSize,
  usage:GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
})
commandEncoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBufSize)
queue.submit([commandEncoder.finish()])

await readBuf.mapAsync(GPUMapMode.READ)
const copyArrayBuffer = readBuf.getMappedRange()
const result  = new Float32Array(copyArrayBuffer.slice(0))
readBuf.unmap()
if(debug) {
    console.log(fromFloat32ArrayMatrix(result,m,n))
}
return fromFloat32ArrayMatrix(result,m,n)
}
if(debug) {
    console.log()
}

function zeros(len) { 
  if(debug) {
    console.log(Array.from({length:len},() => 0) )
}
  return Array.from({length:len},() => 0) 
}
function randMatrix(rows,cols,scale=0.01) { 
if(debug) {
    console.log(Array.from({length:rows},() => Array.from({length:cols},() => (Math.random()*2-1)*scale)) )
}
  return Array.from({length:rows},() => Array.from({length:cols},() => (Math.random()*2-1)*scale)) 
}
function vecAdd(a,b) { return a.map((v,i)=>v+b[i]) }
function vecScale(a,s) { return a.map(v=>v*s) }
function addInPlaceVec(a,b,scale=1) {
  for(let i=0;i<a.length;i++) {
    a[i] += scale * b[i]
  }
}
function addInPlaceMat(a,b,scale=1) {
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[0].length; j++) {
      a[i][j] += b[i][j] * scale
    }
  }
}
async function relu(v) { return v.map(x=>x>0 ? x:0) }
async function geLu(v) { 
  if(debug) {
    console.log(v.map(x=>  0.5 * x * (1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*Math.pow(x,3))))))
}
  return v.map(x=>  0.5 * x * (1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*Math.pow(x,3)))))
}
async function geLuDeriv(v) {
  const c = Math.sqrt(2 / Math.PI)
  return v.map(x => {
    const tanhArg = c * (x + 0.044715 * x**3)
    const tanhVal = Math.tanh(tanhArg)
    const sech2 = 1 - tanhVal**2
    if(debug) {
    console.log(0.5 * (1 + tanhVal) + 0.5 * x * sech2 * c * (1 + 3 * 0.044715 * x**2))
}
    return 0.5 * (1 + tanhVal) + 0.5 * x * sech2 * c * (1 + 3 * 0.044715 * x**2)
  })
}
async function reluDeriv(v) { return v.map(x=>x>0?1:0)}
async function crossEntropyLoss(probs,targetIdx) {
  const maxLogit = Math.max(...probs)
  let sumExp = 0;
  for (let i = 0; i < probs.length; i++) {
    sumExp += Math.exp(probs[i] - maxLogit);
  }
  const logSumExp = maxLogit + Math.log(sumExp);
  if(debug) {
    console.log(logSumExp - probs[targetIdx])
}
  return logSumExp - probs[targetIdx];
}
async function softmax(logits) {
  const max = Math.max(...logits)
  const exps = logits.map(x=>Math.exp(x-max))
  const sum = exps.reduce((s,x)=>s+x,0) + 1e-12
  if(debug) {
    console.log(exps.map(e=>e/sum))
}
  return exps.map(e=>e/sum)
}

let _GPU_DEVICE_ = null
async function ensureGPU() {
  if(!_GPU_DEVICE_) { 
    _GPU_DEVICE_ = await initGPU()
  }
  return _GPU_DEVICE_
}
//----------------------------
//Multi Layer Perceptrons
//----------------------------
class MLP {
  constructor(inputSize,hiddenSize,outputSize) {

    this.inputSize = inputSize
    this.hiddenSize = hiddenSize
    this.outputSize = outputSize

    this.w1 = randMatrix(hiddenSize,inputSize,0.02)
    this.b1 = zeros(hiddenSize)
    this.w2 = randMatrix(outputSize,hiddenSize,0.02)
    this.b2 = zeros(outputSize)
  }

  async forward(x) { 
    const device = await ensureGPU()
    let hpre  
    if(!device) { 
      throw new Error("Device not found")
    }
    else {
      hpre = await matVecMul(device,this.w1,x)
      for(let i=0;i<hpre.length;i++) hpre[i] += this.b1[i]
      
    }
    
    const h = await relu(hpre)

    let logits
    if(device) {
      logits = await matVecMul(device,this.w2,h)
      for(let i=0;i<logits.length;i++) logits[i] += this.b2[i]
    }
    else return console.log("Device not found")
    const probs = await softmax(logits)
    if(debug) {
    console.log(x,hpre,h,logits,probs)
}
    return { x,hpre,h,logits,probs }
  }
  
  async step(x,targetIdx,lr=0.1) {
    const device = await ensureGPU()
    const { hpre,h,probs } = await this.forward(x)
    const dlogits = probs.slice()
    dlogits[targetIdx] -= 1.0

    let gradW2
    if(device) {
      gradW2 = await Outer(device,dlogits,h)
    }
    else return console.log("Device not found")
    const gradB2 = dlogits.slice()
    const dh = Array(this.hiddenSize).fill(0)
    for(let i=0;i<this.hiddenSize;i++) {
      let s = 0
      for(let j=0;j<this.outputSize;j++) s+=this.w2[j][i]*dlogits[j]
      dh[i] = s
    } 
    const drelu = await reluDeriv(hpre)
    const dhpre = dh.map((v,i)=>v*drelu[i])

    let gradW1
    if(device) {
      gradW1 = await Outer(device,dhpre,x)
    }
    else console.log("Device not found")

    const gradb1 = dhpre.slice()

    await addInPlaceMat(this.w2,gradW2,-lr)
    await addInPlaceVec(this.b2,gradB2,-lr)
    await addInPlaceMat(this.w1,gradW1,-lr)
    await addInPlaceVec(this.b1,gradb1,-lr)
    if(debug) {
    await console.log(crossEntropyLoss(probs,targetIdx))
}

    return await crossEntropyLoss(probs,targetIdx)
  }

  async predictTop(x,k=5) {
    const device = await ensureGPU()
    if (!device) throw new Error("Device not found")
    const hpre = (await matVecMul(device, this.w1, x)).map((v, i) => v + this.b1[i])
    const h = (await relu(hpre))
    const logits = await matVecMul(device, this.w2, h)
    const probs = softmax(logits)
    const pairs = probs.map((p, i) => ({ i, p })).sort((a, b) => b.p - a.p)
    if(debug) {
    console.log(pairs.slice(0,k))
}
    return pairs.slice(0, k)
  }
}

class NLP {
  constructor() {
    this.vocab = {}
    this.merges = []
    this.tokenToId = {/* "<pad>": 0 */ };
    this.idToToken = {/* 0: "<pad>" */};
  }
 
  buildVocab(corpus) {
    this.vocab = {};
    for (let line of corpus) {
      const words = line.split(/\s+/).filter(Boolean)
      for (let w of words) {
        const chars = w.split('')
        const token = chars.join(' ') + ' </w>'
        this.vocab[token] = (this.vocab[token] || 0) + 1
      }
    }
  }

  getPairs() {
    const pairs = {}
    for (let word in this.vocab) {
      const symbols = word.split(' ')
      const freq = this.vocab[word]
      for (let i = 0; i < symbols.length - 1; i++) {
        const pair = symbols[i] + ' ' + symbols[i + 1]
        pairs[pair] = (pairs[pair] || 0) + freq
      }
    }
    if(debug) {
    console.log(pairs)
}
    return pairs
  }

  mergeVocab(pair) {
    const [a, b] = pair.split(' ')
    const join = a + b
    const escapedPair = this.escapeRegex(a + ' ' + b)
    const regex = new RegExp(escapedPair, "g")
    const newVocab = {}
    for (let word in this.vocab) {
      const newWord = word.replace(regex, join)
      newVocab[newWord] = this.vocab[word]
    }
    this.vocab = newVocab
  }

  escapeRegex(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  }
  //Train texts and characters
train(corpus, numMerges) {
  this.buildVocab(corpus)
  const chars = new Set(corpus);
  chars.forEach(c => {
    if (!(c in this.tokenToId)) {
      let id = Object.keys(this.tokenToId).length;
      this.tokenToId[c] = id;
      this.idToToken[id] = c;
    }
  })
    for (let i = 0; i < numMerges; i++) {
      const pairs = this.getPairs()
      let bestPair = null
      let bestCount = 0
      for (let p in pairs) {
        if (pairs[p] > bestCount) {
          bestCount = pairs[p]
          bestPair = p
        }
      }
      if (!bestPair || bestCount < 1) break
      this.merges.push(bestPair)
      this.mergeVocab(bestPair)
    }
    this.buildTokenIds()
  }
  
  buildTokenIds() {
  const tokenSet = new Set()
  for (let word in this.vocab) {
    for (let tok of word.split(' ')) {
      if (tok.trim().length > 0) tokenSet.add(tok)
    }
  }
  let i = 1;
  for (let tok of tokenSet) {
    if (!(tok in this.tokenToId)) {
      this.tokenToId[tok] = i
      this.idToToken[i] = tok
      i++
    }
  }
}

  
  applyMergesToWord(word) {
    let symbols = word.split('').concat(['</w>'])
    for (let merge of this.merges) {
      const [a, b] = merge.split(' ')
      if (!a || !b) continue
      let i = 0
      while (i < symbols.length - 1) {
        if (symbols[i] === a && symbols[i + 1] === b) {
          symbols.splice(i, 2, a + b)
        } else {
          i++
        }
      }
    }
    return symbols.filter(s => s && s.trim().length > 0)
  }
  //String to ids
  encode(text) {
  const out = [];
  const words = text.split(/\s+/).filter(Boolean);
  for (let w of words) {
    const toks = this.applyMergesToWord(w);
    for (let t of toks) {
      const id = this.tokenToId[t];
      if (id !== undefined) {
        out.push(id);
      }
    }
  }
  if(debug) {
    console.log(out)
}
  return out;
}
  //Decode tokens(ids)
  decode(ids) {
  let words = []
  let current = ''
  for (let id of ids) {
    if (id === 0) continue;
    const tok = this.idToToken[id]
    if (!tok) continue
    if (tok === '</w>') {
      words.push(current)
      current = ''
    } else {
      current += tok
    }
  }
  if (current) words.push(current)
  return words.join(' ').toLowerCase()
}
}
//-----------------------------
//Embedding Layer
//-----------------------------
class EmbeddingLayer {
  constructor(vocabSize, embedSize) {
    this.vocabSize = vocabSize;
    this.embedSize = embedSize;
    this.tokenEmbed = Array.from({ length: vocabSize }, () =>
      Array.from({ length: embedSize }, () => (Math.random() * 2 - 1) * 0.01)
    )
    this.positionEmbed = Array.from({ length: 512 }, () =>
      Array.from({ length: embedSize }, () => (Math.random() * 2 - 1) * 0.01)
    )
  }

  forward(tokenIds) {
    const seqLength = tokenIds.length
    const output = []
    for (let i = 0; i < seqLength; i++) {
      const tokenVector = this.tokenEmbed[tokenIds[i]]
      const posVector = this.positionEmbed[i]
      if (!tokenVector || !posVector) return new Float32Array(embedSize)

      const combined = tokenVector.map((v, idx) => v + posVector[idx])
      output.push(combined)
    }
    if(debug) {
    console.log(output)
}
    return output
  }
  //(for generate reply)
  async forwardSingle(Id) {
    return (this.forward([Id]))[0]
  }
}
(async () => {
    //log GPU device
    const devObj = await ensureGPU()
    console.log(devObj)
})()
//-----------------------------
//Multi Head Attention
//-----------------------------
class MHA {
  constructor(device,embedSize,numHeads) {
    this.device = device
    this.embedSize = embedSize
    this.numHeads = numHeads
    this.headDim = embedSize/numHeads
    
    this.WQ = randMatrix(embedSize,embedSize)
    this.WK = randMatrix(embedSize,embedSize)
    this.Wv = randMatrix(embedSize,embedSize)
    this.Wo = randMatrix(embedSize,embedSize)
  }

  async splitHeads(X) {
  const H = [];
  const matrixX = (typeof X[0] === 'number') ? [X] : X;

  for(let h=0; h<this.numHeads; h++) {
    H.push(
      matrixX.map(row => {
        return row.slice ? row.slice(h*this.headDim, (h+1)*this.headDim) : 
                           Array.from(row).slice(h*this.headDim, (h+1)*this.headDim);
      })
    )
  }
  if(debug) {
    console.log(H)
}
  return H;
}
  concatHeads(H) {
    return H[0].map((_,r)=>
      H.map(h=>h[r]).flat()
    )
  }

  scaleDot(Q,K) {
    K = Array(K)
    const KT = K[0].map((_,c)=>K.map(r=>r[c]))
    const S = Q.map(rowQ=>KT.map(
      rowK=>rowQ.reduce((s,v,i)=>s+v*rowK[i],0)
    ))
    const scale = Math.sqrt(this.headDim)
    return S.map(r=>r.map(v=>v/scale))
  }

  async forward(x) {
  const Q = await matMul(this.device, x,this.WQ)
  const K = await matMul(this.device, x,this.WK)
  const V = await matMul(this.device, x,this.Wv)

  const Qh = await this.splitHeads(Q)
  const Kh = await this.splitHeads(K)
  const Vh = await this.splitHeads(V)

  const headOutputs = []
  for(let h=0; h<this.numHeads; h++) {
    const q = Qh[h]
    const k = Kh[h]
    const v = Vh[h]
    
    const score = this.scaleDot(q, k)
    const soft = await softmax(score[0])
    const out = await matMul(this.device, [soft], v) 
    headOutputs.push(out)
  }
  
  const concat = this.concatHeads(headOutputs)
  const finalOut = await matMul(this.device, concat,this.Wo) 
if(debug) {
    console.log("MHA FİNAL",Q,K,V,finalOut)
}
  return finalOut
}
}
//--------------------------
//Layer normalization
//--------------------------
class LayerNorm {
  constructor(embedSize,{eps=1e-5,affline=true}={}) {
    this.embedSize = embedSize
    this.eps=eps
    this.affline =affline

    this.gamma = new Float32Array(embedSize)
    this.beta = new Float32Array(embedSize)
    for(let i=0;i<embedSize;i++) {
      this.gamma[i] = 1.0
      this.beta[i] = 0.0
    }
  }

  forwardVector(x) {
    const n = this.embedSize
    const xv = (x instanceof Float32Array) ? x:Float32Array.from(x)

    let sum =0
    for(let i=0;i<n;i++) sum += xv[i]
    const mean = sum/n

    let sq = 0
    for(let i=0;i<n;i++) {
      const d=xv[i]-mean
      sq+=d*d
    }
    const varr = sq/n
    const denom = 1/Math.sqrt(varr+this.eps)
    const out= new Float32Array(n)
    
    for(let i=0;i<n;i++) {
      const norm = (xv[i]-mean) *denom
      out[i] = (this.affline ? (this.gamma[i] * norm+this.beta[i]):norm)
    }
    if(debug) {
    console.log(out)
}
    return out
  }

  forwardBatch(X) {
    return X.map(x=>this.forwardVector(x))
  }

  forward(X) {
    if(!X) return null
    if(X instanceof Float32Array || Array.isArray(X)&& typeof X[0] === "number")  {
      return this.forwardVector(X)
    }
    return this.forwardBatch(X)
  }
}
//Load dataset in fetch
async function loadDataset(file) {
  const response = await fetch(file)
  const arrayBuffer = await response.arrayBuffer()
  const decoder = new TextDecoder("utf-8")
  const text = decoder.decode(arrayBuffer)
  const lines = text.split("\n").map(x => x.trim()).filter(Boolean)
  return lines
}
function shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}
//----------------------------
//Training loop
//----------------------------
async function TrainModel(device,mlp,embed,nlp,dataset,numMerges = 500,epochs = 3,lr = 0.01,maxContextLength = 64 ) {
  if (!device) throw new Error("GPU device required")

  if (dataset instanceof Promise) dataset = await dataset
 


  if (Object.keys(nlp.tokenToId || {}).length === 0) {
    if (typeof nlp.train === "function") {
      nlp.train(dataset, numMerges)
    }
  }

  const IN = mlp.inputSize || mlp.inSize || mlp.input_size || (mlp.w1 && mlp.w1.length ? mlp.w1[0].length : null)
  const H  = mlp.hiddenSize || mlp.hiddenSize || (mlp.w1 ? mlp.w1.length : null);
  const V  = mlp.outputSize || mlp.vocabSize || mlp.vocab_size || (mlp.w2 && mlp.w2.length ? mlp.w2[0].length : null)

  if (!IN || !H || !V) throw new Error("mlp must provide inputSize, hiddenSize, outputSize (or compatible w1/w2 shape)")


  function ensureRowFloat32Array(matrix, rows, cols) {
    const out = new Array(rows)
    for (let r = 0; r < rows; r++) {
      const row = matrix[r]
      if (row instanceof Float32Array) {
        if (row.length !== cols) {

          const tmp = new Float32Array(cols)
          tmp.set(row.subarray(0, Math.min(row.length, cols)))
          out[r] = tmp
        } else {
          out[r] = row
        }
      } else if (Array.isArray(row)) {
        const tmp = new Float32Array(cols)
        for (let c = 0; c < Math.min(row.length, cols); c++) tmp[c] = row[c]
        out[r] = tmp
      } else {
  
        out[r] = new Float32Array(cols)
      }
    }
    return out
  }

  let w1_by_input;
  if (mlp.w1 && Array.isArray(mlp.w1) && mlp.w1.length === H && (mlp.w1[0].length === IN || mlp.w1[0] instanceof Float32Array && mlp.w1[0].length === IN)) {

    const tmp = new Array(IN);
    for (let i = 0; i < IN; i++) {
      const row = new Float32Array(H)
      for (let h = 0; h < H; h++) row[h] = mlp.w1[h][i] ?? 0.0
      tmp[i] = row
    }
    w1_by_input = tmp
  } else if (mlp.w1 && Array.isArray(mlp.w1) && mlp.w1.length === IN) {

    w1_by_input = ensureRowFloat32Array(mlp.w1, IN, H)
  } else {

    w1_by_input = []
    for (let i = 0; i < IN; i++) w1_by_input.push(new Float32Array(H))
  }

  const b1_arr = (mlp.b1 instanceof Float32Array) ? mlp.b1 : new Float32Array((mlp.b1 || []).length || H)

  let w2_by_hidden;
  if (mlp.w2 && Array.isArray(mlp.w2) && mlp.w2.length === H) {
    w2_by_hidden = ensureRowFloat32Array(mlp.w2, H, V)
  } else {
 
    w2_by_hidden = [];
    for (let h = 0; h < H; h++) w2_by_hidden.push(new Float32Array(V))
  }
  const b2_arr = (mlp.b2 instanceof Float32Array) ? mlp.b2 : new Float32Array((mlp.b2 || []).length || V)


  const W1_len = IN * H
  const b1_len = H
  const W2_len = H * V
  const b2_len = V
  const totalLen = W1_len + b1_len + W2_len + b2_len
  const mlpFlat = new Float32Array(totalLen)
  
  let offset = 0

  for (let i = 0; i < IN; i++) {
    const row = w1_by_input[i]
    for (let h = 0; h < H; h++) {
      mlpFlat[offset++] = row[h] ?? 0.0
    }
  }

  for (let h = 0; h < H; h++) mlpFlat[offset++] = b1_arr[h] ?? 0.0

  for (let h = 0; h < H; h++) {
    const row = w2_by_hidden[h]
    for (let v = 0; v < V; v++) {
      mlpFlat[offset++] = row[v] ?? 0.0
    }
  }

  for (let v = 0; v < V; v++) mlpFlat[offset++] = b2_arr[v] ?? 0.0

  const mlpBufferSize = mlpFlat.byteLength;

  const mlpBuffer = device.createBuffer({
    size: ((mlpBufferSize + 255) & ~255),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  })
  device.queue.writeBuffer(mlpBuffer, 0, mlpFlat)


  const xBufferSize = IN * Float32Array.BYTES_PER_ELEMENT;
  const xBuffer = device.createBuffer({
  size: Math.max(IN * 4, 16),
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});
const targetBuffer = device.createBuffer({
  size: 16,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});
  const lossBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  })
  const lossReadBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  })
  const cfgBuffer = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  })

  {
  const ab = new ArrayBuffer(32);
  const dv = new DataView(ab);
  dv.setUint32(0, IN, true);
  dv.setUint32(4, H, true);
  dv.setUint32(8, V, true);
  dv.setFloat32(12, lr, true);
  dv.setFloat32(16, 0.0, true);
  device.queue.writeBuffer(cfgBuffer, 0, ab);
}
  const xData = new Float32Array(IN)
  const targetVal = new Uint32Array([0])
  device.queue.writeBuffer(xBuffer, 0, xData)
  device.queue.writeBuffer(targetBuffer, 0, targetVal)
  const shaderCode = `
struct ModelConfig {
    inSize: u32,
    hiddenSize: u32,
    vocabSize: u32,
    lr: f32,
    pad: f32,
};

@group(0) @binding(1) var<storage, read_write> W1: array<f32>;
@group(0) @binding(2) var<storage, read_write> b1: array<f32>;
@group(0) @binding(3) var<storage, read_write> W2: array<f32>;
@group(0) @binding(4) var<storage, read_write> b2: array<f32>;
@group(0) @binding(5) var<storage, read>       x: array<f32>;
@group(0) @binding(6) var<storage, read>       targetVal: array<u32>;
@group(0) @binding(7) var<uniform>             config: ModelConfig;
@group(0) @binding(8) var<storage, read_write> lossOut: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }

    let IN = config.inSize;
    let H  = config.hiddenSize;
    let V  = config.vocabSize;
    let LR = config.lr;

   
    var hidden: array<f32, 1024>;
    var hiddenMask: array<f32, 1024>;
    var logits: array<f32, 1024>;
    var probs: array<f32, 1024>;
    var gradLogits: array<f32, 1024>;
    var gradHidden: array<f32, 1024>;

    // Forward hidden
    for (var h = 0u; h < H; h = h + 1u) {
        var s: f32 = b1[h];
        for (var i = 0u; i < IN; i = i + 1u) {
            let wi = i * H + h;
            s = s + x[i] * W1[wi];
        }
        if (s > 0.0) {
            hidden[h] = s;
            hiddenMask[h] = 1.0;
        } else {
            hidden[h] = 0.0;
            hiddenMask[h] = 0.0;
        }
    }

    // Forward output
    var maxLogit: f32 = -1e30;
    for (var v = 0u; v < V; v = v + 1u) {
        var s: f32 = b2[v];
        for (var h = 0u; h < H; h = h + 1u) {
            let wi = h * V + v;
            s = s + hidden[h] * W2[wi];
        }
        logits[v] = s;
        if (s > maxLogit) { maxLogit = s; }
    }

    // Softmax
    var expSum: f32 = 0.0;
    for (var v = 0u; v < V; v = v + 1u) {
        let e = exp(logits[v] - maxLogit);
        probs[v] = e;
        expSum = max(expSum+e, 1e-9);
    }
    for (var v = 0u; v < V; v = v + 1u) {
        probs[v] = probs[v] / expSum;
    }

    // Loss and gradient output
    let t = targetVal[0];
    for (var v = 0u; v < V; v = v + 1u) {
        gradLogits[v] = probs[v];
    }
    gradLogits[t] = gradLogits[t] - 1.0;
    let loss = -log(probs[t] + 1e-9);
    lossOut[0] = loss;

    // Backprop hidden
    for (var h = 0u; h < H; h = h + 1u) {
        var s: f32 = 0.0;
        for (var v = 0u; v < V; v = v + 1u) {
            let wi = h * V + v;
            s = s + gradLogits[v] * W2[wi];
        }
        gradHidden[h] = s * hiddenMask[h];
    }

    // Update W2, b2
    for (var v = 0u; v < V; v = v + 1u) {
        b2[v] = b2[v] - LR * gradLogits[v];
        for (var h = 0u; h < H; h = h + 1u) {
            let wi = h * V + v;
            W2[wi] = W2[wi] - LR * gradLogits[v] * hidden[h];
        }
    }

    // Update W1, b1
    for (var h = 0u; h < H; h = h + 1u) {
        b1[h] = b1[h] - LR * gradHidden[h];
        for (var i = 0u; i < IN; i = i + 1u) {
            let wi = i * H + h;
            W1[wi] = W1[wi] - LR * gradHidden[h] * x[i];
        }
    }
}


`

  const module = device.createShaderModule({ code: shaderCode });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" }
  })

  const bindGroupLayout = pipeline.getBindGroupLayout(0);
  

function sliceFloat32ToBuffer(startOffsetBytes, lengthFloats, bindingName = "unknown") {
  const actualLength = Math.max(lengthFloats, 1);
  const arr = new Float32Array(actualLength);
  
  const startIndex = startOffsetBytes / 4;
  for (let i = 0; i < lengthFloats; i++) {
    arr[i] = mlpFlat[startIndex + i] ?? 0.0;
  }

  const buf = device.createBuffer({
    size: Math.max(arr.byteLength, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(buf, 0, arr);
  return buf;
}



  const w1Bytes = W1_len * 4
  const b1Bytes = b1_len * 4
  const w2Bytes = W2_len * 4
  const b2Bytes = b2_len * 4

  let byteOffset = 0
  const W1_buf = sliceFloat32ToBuffer(byteOffset, W1_len, "W1"); byteOffset += w1Bytes
  const b1_buf = sliceFloat32ToBuffer(byteOffset, b1_len, "b1"); byteOffset += b1Bytes
  const W2_buf = sliceFloat32ToBuffer(byteOffset, W2_len, "W2"); byteOffset += w2Bytes
  const b2_buf = sliceFloat32ToBuffer(byteOffset, b2_len, "b2"); byteOffset += b2Bytes
console.log("Sizes:", {W1: W1_len, b1: b1_len, IN: IN, V: V});
const bindGroupFinal = device.createBindGroup({
  layout: bindGroupLayout,
  entries: [
    { binding: 1, resource: { buffer: W1_buf } },
    { binding: 2, resource: { buffer: b1_buf } },
    { binding: 3, resource: { buffer: W2_buf } },
    { binding: 4, resource: { buffer: b2_buf } },
    { binding: 5, resource: { buffer: xBuffer } },
    { binding: 6, resource: { buffer: targetBuffer } },
    { binding: 7, resource: { buffer: cfgBuffer } },
    { binding: 8, resource: { buffer: lossBuffer } },
  ]
});
  function dispatchSingle(pass) {
    if (typeof pass.dispatchWorkgroups === "function") pass.dispatchWorkgroups(1)
    else if (typeof pass.dispatch === "function") pass.dispatch(1)
    else if (typeof pass.dispatchWorkgroup === "function") pass.dispatchWorkgroup(1)
    else pass.dispatch(1)
  }


  let stepCount = 0
  let totalLoss = 0

  for (let epoch = 0; epoch < epochs; epoch++) {
    for (let line of dataset) {
      const words = line.split(/\s+/).filter(Boolean)
      if (words.length < 1) continue

      const seqIds = []
      for (const w of words) {
        const ids = nlp.encode(w)
        for (const id of ids) seqIds.push(id)
      }
      if (seqIds.length < 2) continue

      for (let i = 0; i < seqIds.length - 1; i++) {
        const startIdx = Math.max(0, i + 1 - maxContextLength)
        const contextIds = seqIds.slice(startIdx, i + 1)
        const targetId = seqIds[i + 1]

        const E = embed.forward(contextIds)
        const x = new Float32Array(IN)
        for (let t = 0; t < E.length; t++) {
          const row = E[t]
          for (let j = 0; j < IN; j++) {
            x[j] += row[j] ?? 0.0
          }
        }
        const denom = Math.max(1, E.length)
        for (let j = 0; j < IN; j++) x[j] /= denom


        device.queue.writeBuffer(xBuffer, 0, x.buffer ? x : new Float32Array(x))
        device.queue.writeBuffer(targetBuffer, 0, new Uint32Array([targetId]))


        const encoder = device.createCommandEncoder()
        const pass = encoder.beginComputePass()
        pass.setPipeline(pipeline)
        pass.setBindGroup(0, bindGroupFinal)
        dispatchSingle(pass)
        pass.end()

        encoder.copyBufferToBuffer(lossBuffer, 0, lossReadBuffer, 0, 4)
        device.queue.submit([encoder.finish()])

        await lossReadBuffer.mapAsync(GPUMapMode.READ)
        const dv = new Float32Array(lossReadBuffer.getMappedRange().slice(0))
        const loss = dv[0]
        lossReadBuffer.unmap()

        totalLoss += Number(loss) || 0
        stepCount++

        if (stepCount % 100 === 0) {
          console.log(`Epoch ${epoch + 1}/${epochs} Step ${stepCount} Loss: ${loss}`)
        }
      } 
    } 

    console.log(`Epoch ${epoch + 1} ended | Avarage Loss: ${(totalLoss / Math.max(1, stepCount)).toFixed(4)}`)
  }
  const readBuf = device.createBuffer({
    size: ((mlpBufferSize + 255) & ~255),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  })

  const finalEncoder = device.createCommandEncoder()
  finalEncoder.copyBufferToBuffer(W1_buf, 0, readBuf, 0, w1Bytes)
  finalEncoder.copyBufferToBuffer(b1_buf, 0, readBuf, w1Bytes, b1Bytes)
  finalEncoder.copyBufferToBuffer(W2_buf, 0, readBuf, w1Bytes + b1Bytes, w2Bytes)
  finalEncoder.copyBufferToBuffer(b2_buf, 0, readBuf, w1Bytes + b1Bytes + w2Bytes, b2Bytes)
  device.queue.submit([finalEncoder.finish()])

  await readBuf.mapAsync(GPUMapMode.READ)
  const mapped = new Float32Array(readBuf.getMappedRange().slice(0))
  readBuf.unmap()

  let idx = 0;
  const newW1_by_input = []
  for (let i = 0; i < IN; i++) {
    const row = new Float32Array(H)
    for (let h = 0; h < H; h++) {
      row[h] = mapped[idx++]
    }
    newW1_by_input.push(row)
  }
  const newB1 = new Float32Array(H)
  for (let h = 0; h < H; h++) newB1[h] = mapped[idx++]

  const newW2_by_hidden = []
  for (let h = 0; h < H; h++) {
    const row = new Float32Array(V)
    for (let v = 0; v < V; v++) row[v] = mapped[idx++]
    newW2_by_hidden.push(row)
  }
  const newB2 = new Float32Array(V)
  for (let v = 0; v < V; v++) newB2[v] = mapped[idx++]

  const finalW1 = []
  for (let h = 0; h < H; h++) {
    const row = new Float32Array(IN)
    for (let i = 0; i < IN; i++) row[i] = newW1_by_input[i][h]
    finalW1.push(row)
  }

  mlp.w1 = finalW1
  mlp.b1 = Array.from(newB1)
  mlp.w2 = newW2_by_hidden
  mlp.b2 = Array.from(newB2)


  try {
    if (W1_buf.destroy) W1_buf.destroy()
    if (b1_buf.destroy) b1_buf.destroy()
    if (W2_buf.destroy) W2_buf.destroy()
    if (b2_buf.destroy) b2_buf.destroy()
    if (mlpBuffer.destroy) mlpBuffer.destroy()
    if (xBuffer.destroy) xBuffer.destroy()
    if (targetBuffer.destroy) targetBuffer.destroy()
    if (lossBuffer.destroy) lossBuffer.destroy()
    if (lossReadBuffer.destroy) lossReadBuffer.destroy()
    if (readBuf.destroy) readBuf.destroy()
    if (cfgBuffer.destroy) cfgBuffer.destroy()
  } catch (e) {
   
  }
  if(debug) {
    console.log(mlp)
}

  return {mlp,nlp,embed}
}

function normalizeProbs(probs) {
  const clipped = probs.map(p => (isFinite(p) && p > 0 ? p : 0))
  const sum = clipped.reduce((s, v) => s + v, 0) + 1e-12
  return clipped.map(p => p / sum)
}
function getSingleEmbedding(embedding, tokenId) {
  if (typeof embedding.forwardSingle === "function") {
    return embedding.forwardSingle(tokenId)
  }
  const out = embedding.forward([tokenId])
  return out[0]
}
function applyTemperature(logits,temperature) {
  if(temperature===1.0) return logits
  return logits.map(v=>v/temperature)
}
function TopK(probs,k) {
  probs = normalizeProbs(probs)
  const sorted=probs.map((p,i)=>({p,i})).sort((a,b)=>b.p-a.p)
  const top=sorted.slice(0,k)
  let sum = top.reduce((a,b)=>a+b.p,0)
  let r=Math.random()*sum
  for(let t of top) {
    r-=t.p
    if(r<=0) return t.i
  }
  probs[0] = 0
  return top[top.length-1].i
}
function TopP(probs,p=0.9) {
  probs = normalizeProbs(probs)
  const sorted=probs.map((p,i)=>({p,i})).sort((a,b)=>b.p-a.p)
  let cumulative = 0
  const top = []
  for(let t of sorted) {
    top.push(t)
    cumulative += t.p
    if(cumulative >=p) break
  }
  let sum = top.reduce((a,b)=>a+b.p,0)
  let r = Math.random()*sum
  for(let t of top) {
    r -= t.p
    if(r<=0) return t.i
  }
  return top[top.length-1].i
}
function addVec(a,b) {
  const n = a.length
  const r = new Float32Array(n)
  for(let i=0;i<n;i++) r[i] = a[i]+b[i]
  return r
}
async function generateReply(mha, mlp, layernorm, nlp, embedding, prompt, {
  maxTokens = 40,
  temperature = 0.9,
  topK = 40,
  topP = 1.0,
  repeatPenalty = 2.5
} = {}) {

  const device = await ensureGPU()
  if (!device) return "Device not found"
  

  let text = String(prompt).trim()
  if (text === undefined) return ""

  const usedCounts = {}

  for (let step = 0; step < maxTokens; step++) {
    const ids = nlp.encode(text).slice(-16)
    if (!ids || ids.length === 0) break
    const lastId = ids[ids.length - 1]
  if (lastId === undefined || lastId < 0) break;
  const x = await embedding.forward(ids)
    
    const xNorm1 = await layernorm.forward(x)
    const mhaOut = await mha.forward(xNorm1)
    const x1 = addVec(x, mhaOut)
 
    const xNorm2 = await layernorm.forward(x1)
    const lastVec = xNorm2[xNorm2.length - 1]
    const res = await mlp.forward(lastVec)
    let logits = res.logits || null
    let probs = res.probs || null

 
    if (logits) {
      for (const tidStr in usedCounts) {
        const tid = Number(tidStr);
        if (tid >= 0 && tid < logits.length) {
          if (logits[tid] > 0) {
             logits[tid] /= repeatPenalty;
        } else {
        logits[tid] *= repeatPenalty;
      }
    }
  }
  const scaledLogits = applyTemperature(logits, temperature);
  probs = softmax(scaledLogits);
} else if (probs) {

    } else {
      break
    }

    if (!probs || probs.length === 0) break
    let nextId
    if (topK > 0 && topK < probs.length) nextId = TopK(probs, topK)
    else if (topP < 1.0) nextId = TopP(probs, topP)
    else {
      let best = 0, bestP = -Infinity
      for (let i = 0; i < probs.length; i++) {
        if (probs[i] > bestP) {
          bestP = probs[i]
          best = i
        }
      }
      nextId = best
    }

    usedCounts[nextId] = (usedCounts[nextId] || 0) + 1
    const tok = nlp.idToToken[nextId]
    if (!tok) break

    const clean = tok.replace("</w>", "")
    if (text.length === 0) text = clean
    else text += " " + clean
  }

  return text
}

async function StartModel(dataset, numMerges, EmbedSize, hiddenSize, epochs=3, lr=0.01) {
  const device = await ensureGPU();
  const data = await loadDataset(dataset);
  const nlp = new NLP();
  nlp.train(data,numMerges)
  const vocabSize = Object.keys(nlp.tokenToId).length;
  if (vocabSize === 0) {
    throw new Error("Vocab is not created.");
  }
  console.log("Vocab Size:", vocabSize)
  const embed = new EmbeddingLayer(vocabSize, EmbedSize);
  const mlp = new MLP(EmbedSize, hiddenSize, vocabSize);
  await TrainModel(device, mlp, embed, nlp, data, numMerges, epochs, lr, 64);
  return { mlp, nlp, embed };
}



(async () => {
  const device = await ensureGPU();
  const embedSize = 128
  const hiddenSize = 256
  const trainedModel = await StartModel("http://127.0.0.1:5500/datasets/en_daily_dialog_ds.txt", 500, embedSize, hiddenSize, 3, 0.0003);
  const usrprompt = "Hello";
  debug = false
  const mha = new MHA(device,embedSize,3)
  const layernorm = new LayerNorm(embedSize)
  const response = await generateReply(
    mha,
    trainedModel.mlp, 
    layernorm, 
    trainedModel.nlp, 
    trainedModel.embed, 
    usrprompt
  );
  
  console.log("Aira:", response);

})();

