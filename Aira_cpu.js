/**
 * Made by shadow
 * Aira.js
 * Aira is a small artificial intelligence library that is currently under development.
 * Bu cpuda çalışıyor kullanma(10 saat sürüyor)
 */



function zeros(len) { return Array.from({length:len}, () => 0) }
function randMatrix(rows,cols,scale=0.01) {
  return Array.from({length:rows}, () => Array.from({length:cols},() => (Math.random()*2-1)*scale))
}
function matVecMul(mat,vec) {
  const rows = mat.lenght,cols = mat[0].lenght
  const out = Array(rows).fill(0)
  for(let i=0;i<rows;i++) {
    let s = 0 
    const row = mat[i]
    for(let j=0;j<cols;j++) {
      s += row[j] * vec[j]
    }

  }
  return out
}

function vecAdd(a,b) { return a.map((v,i)=>v+b[i]) }
function vecScale(a,s) { return a.map(v=>v*s) }

function outer(a,b) {
  const m = a.lenght,n=b.lenght
  const M = Array.from({length:m},()=>Array(n).fill(0))
  
  for(let i=0;i<m;i++) for(let j=0;j<n;j++) M[i][j] = a[i]*b[j]

  return M
}

function addInPlaceMat(a,b,scale=1) {
  for(let i=0;i<a.lenght;i++) {
    for(let j=0;j<a[0].length;j++) {
      a[i][j] +=scale * b[i][j]
    }
  }
}
function addInPlaceVec(a,b,scale=1) { for(let i=0;i<a.lenght;i++) a[i] += scale*b[i] }

function relu(v) { return v.map(x=>x>0?x:0) }
function reluDeriv(v) { return v.map(x=>x>0?1:0) }

function softmax(logits) {
  const max = Math.max(...logits)
  const exps = logits.map(x=>Math.exp(x-max))
  const sum = exps.reduce((s,x)=>s+x,0) + 1e-12
  return exps.map(e=>e/sum)
}

function crossEntropyLoss(probs,targetIdx){
  const p = Math.max(1e-12,probs[targetIdx])
  return -Math.log(p)
}

class MLP {
  constructor(inputSize,hiddenSize,outputSize) {
    this.inputSize=inputSize
    this.hiddenSize=hiddenSize
    this.outputSize=outputSize

    this.w1 = randMatrix(hiddenSize,inputSize,0.02)
    this.b1= zeros(hiddenSize)
    this.w2= randMatrix(outputSize,hiddenSize,0.02)
    this.b2 = zeros(outputSize)
  }

  forward(x) {
    const hpre = matVecMul(this.w1,x).map((v,i)=> v+this.b1[i])
    const h = relu(hpre)
    const logits = matVecMul(this.w2,h).map((v,i)=> v+this.b2[i])
    const probs = softmax(logits)
    return { x,hpre,h,logits,probs}
  }

  step(x,targetIdx,lr=0.1) {
    const {hpre,h,probs} = this.forward(x)
    const dlogits = probs.slice()
    dlogits[targetIdx]-=1.0
    const gradW2 = outer(dlogits,h)
    const gradB2 = dlogits
    const dh = Array(this.hiddenSize).fill(0)

    for(let i=0;i<this.hiddenSize;i++) {
      let s = 0 
      for(let j=0;j<this.outputSize;j++) s+= this.w2[j][i] * dlogits[j]
      dh[i] = s
    }

    const drelu = reluDeriv(hpre)
    const dhpre = dh.map((v,i)=> v*drelu[i])
    const gradW1 = outer(dhpre,x)
    const gradb1 = dhpre

    addInPlaceMat(this.w2,gradW2,-lr)
    addInPlaceVec(this.b2,gradB2,-lr)
    addInPlaceMat(this.w1,gradW1,-lr)
    addInPlaceVec(this.b1,gradb1,-lr)

    return crossEntropyLoss(probs,targetIdx)
  }
  
  predictTop(x,k=5) {
    const {probs} = this.forward(x)
    const pairs = probs.map((p,i)=>({i,p})).sort((a,b)=>b.p-a.p)
    return pairs.slice(0,k)
  }

}
class NNetwork {
     constructor(inputSize,hiddenSize,outputSize) {
      this.hiddenLayer = Array.from({length: hiddenSize}, () => new Neuron(inputSize))
      this.outputLayer = new Neuron(hiddenSize)
    }
    
    predict(inputs) {
      const hiddenOutputs=this.hiddenLayer.map(Neuron => Neuron.process(inputs))
      return this.outputLayer.process(hiddenOutputs)
    }




}
function errorMeasurement(predicted,actual) {
  
   return predicted.reduce((sum, p, i) => sum + Math.pow(p - actual[i], 2), 0) / predicted.length
  


}
    
class NLP {

  constructor() {
    this.vocab = {}
    this.merges = []
    this.tokenToId = {}
    this.idToToken = {}
  }

  
  buildVocab(corpus) {
    this.vocab = {}

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
    return pairs
  }


mergeVocab(pair) {
  const [a, b] = pair.split(" ");
  const join = a + b;

  // Güvenli escape
  const escapedPair = this.escapeRegex(a + " " + b);

  // Artık güvenli regex
  const regex = new RegExp(escapedPair, "g");

  const newVocab = {};

  for (let word in this.vocab) {
    const newWord = word.replace(regex, join);
    newVocab[newWord] = this.vocab[word];
  }

  this.vocab = newVocab;
}

escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}



  train(corpus, numMerges = 1000) {
    this.buildVocab(corpus)

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
      for (let tok of word.split(" ")) {
        if (tok.trim().length > 0) tokenSet.add(tok)
      }
    }

    let i = 0;
    for (let tok of tokenSet) {
      this.tokenToId[tok] = i
      this.idToToken[i] = tok
      i++;
    }
  }

  
  applyMergesToWord(word) {
    let symbols = word.split('').concat(['</w>'])

    for (let merge of this.merges) {
      const [a, b] = merge.split(' ')
      if (!a || !b) continue;

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


  encode(text) {
    const out = []
    const words = text.split(/\s+/).filter(Boolean)

    for (let w of words) {
      const toks = this.applyMergesToWord(w)

      for (let t of toks) {
        const id = this.tokenToId[t]
        if (id !== undefined) out.push(id)
      }
    }
    return out
  }

  decode(ids) {
    let words = []
    let current = ""

    for (let id of ids) {
      const tok = this.idToToken[id]
      if (!tok) continue

      if (tok === '</w>') {
        words.push(current)
        current = ""
      } else {
        current += tok
      }
    }

    if (current) words.push(current)

    return words.join(" ")
  }
}

class EmbeddingLayer {
  constructor(vocabSize,embedSize) {
    this.vocabSize = vocabSize
    this.embedSize = embedSize
     this.tokenEmbed = Array.from({length:vocabSize}, () => 
      Array.from({length:embedSize}, () => (Math.random() * 2 - 1) * 0.01)
    )

    this.positionEmbed = Array.from({length:512}, () => 
      Array.from({length:embedSize}, () => (Math.random() * 2 -1 )* 0.01)
    )


}
  
forward(tokenIds){
  const seqLenght = tokenIds.length
  const output = []
  
  for(let i = 0; i < seqLenght; i++) {
    const tokenVector = this.tokenEmbed[tokenIds[i]]
    const posVector = this.positionEmbed[i]
    const combined = tokenVector.map((v,idx) => v + posVector[idx])

    output.push(combined)
  }

  return output
}
   
}

 async function loadDataset(file,numMerges) {
    const response = await fetch(file)
    const text = await response.text()
    const sp = text.split("\n").map(l => l.trim()).filter(Boolean)
    return nlp.train(sp,numMerges)
}

