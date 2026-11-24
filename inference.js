/* Inference for the saved TakeMeter model. BLAKE2b uses a 4-byte output. */
const IV = [
  0x6a09e667f3bcc908n, 0xbb67ae8584caa73bn, 0x3c6ef372fe94f82bn, 0xa54ff53a5f1d36f1n,
  0x510e527fade682d1n, 0x9b05688c2b3e6c1fn, 0x1f83d9abfb41bd6bn, 0x5be0cd19137e2179n
];
const SIGMA = [
  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
  [14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3],
  [11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4],
  [7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8],
  [9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13],
  [2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9],
  [12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11],
  [13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10],
  [6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5],
  [10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0]
];
const MASK = (1n << 64n) - 1n;
const rot = (x, n) => (x >> n) | ((x << (64n - n)) & MASK);
const encoder = new TextEncoder();

function blakeFour(text) {
  const bytes = encoder.encode(text);
  const h = IV.slice();
  h[0] ^= 0x01010004n;
  for (let offset = 0; ; offset += 128) {
    const block = new Uint8Array(128);
    block.set(bytes.subarray(offset, offset + 128));
    const view = new DataView(block.buffer);
    const m = Array.from({length:16}, (_, i) => view.getBigUint64(i * 8, true));
    const v = h.concat(IV);
    const last = offset + 128 >= bytes.length;
    v[12] ^= BigInt(Math.min(bytes.length, offset + 128));
    if (last) v[14] ^= MASK;
    function mix(a, b, c, d, x, y) {
      v[a] = (v[a] + v[b] + x) & MASK;
      v[d] = rot(v[d] ^ v[a], 32n);
      v[c] = (v[c] + v[d]) & MASK;
      v[b] = rot(v[b] ^ v[c], 24n);
      v[a] = (v[a] + v[b] + y) & MASK;
      v[d] = rot(v[d] ^ v[a], 16n);
      v[c] = (v[c] + v[d]) & MASK;
      v[b] = rot(v[b] ^ v[c], 63n);
    }
    for (let r = 0; r < 12; r++) {
      const s = SIGMA[r % 10];
      mix(0,4,8,12,m[s[0]],m[s[1]]);
      mix(1,5,9,13,m[s[2]],m[s[3]]);
      mix(2,6,10,14,m[s[4]],m[s[5]]);
      mix(3,7,11,15,m[s[6]],m[s[7]]);
      mix(0,5,10,15,m[s[8]],m[s[9]]);
      mix(1,6,11,12,m[s[10]],m[s[11]]);
      mix(2,7,8,13,m[s[12]],m[s[13]]);
      mix(3,4,9,14,m[s[14]],m[s[15]]);
    }
    for (let i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
    if (last) return Number(h[0] & 0xffffffffn);
  }
}

function decodeFloats(base64) {
  const raw = atob(base64);
  const bytes = Uint8Array.from(raw, c => c.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  return Float32Array.from({length:bytes.length / 4}, (_, i) => view.getFloat32(i * 4, true));
}

function tokenize(text) {
  return (text.normalize('NFKC').toLowerCase().match(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)?/gu) || [])
    .slice(0,220).map(word => word.replace('’', "'"));
}

function featuresFor(text, model) {
  const words = tokenize(text);
  if (!words.length) throw new Error('Enter an answer with at least one word.');
  const semantic = new Float64Array(300);
  let seen = 0;
  for (const word of words) {
    const index = model.vocabulary.get(word);
    if (index === undefined) continue;
    seen++;
    for (let i = 0; i < semantic.length; i++) semantic[i] += model.embedded[index * 300 + i];
  }
  const ngrams = new Set(words);
  for (let i = 1; i < words.length; i++) ngrams.add(words[i - 1] + ' ' + words[i]);
  const lexical = new Float64Array(768);
  for (const gram of ngrams) {
    const hash = blakeFour(gram);
    lexical[hash % 768] += hash >= 0x80000000 ? 1 : -1;
  }
  const norm = Math.max(1, Math.hypot(...lexical));
  const count = expression => (text.match(expression) || []).length;
  const surface = [
    words.length, count(/[.!?](?:\s|$)/g), count(/\p{Nd}+/gu),
    count(/["“”]/g), count(/:/g), count(/\(/g), count(/\b[A-Z]{2,}\b/g),
    count(/\b\d{1,2}:\d\d(?::\d\d)?\b/g),
    count(/\b(?:wikipedia|imdb|interview|script|screenplay|quoted|transcript|noveli[sz]ation|commentary|credits|according to)\b/gi),
    count(/\b(?:because|therefore|thus|which means|which is why|so that|as a result)\b/gi)
  ].map((n,i) => Math.min(5, Math.max(-5, (Math.log1p(n) - model.center[i]) / model.scale[i])));
  return [...semantic.map(value => value * 4 / Math.max(seen, 1)),
    ...lexical.map(value => value * 1.7 / norm), ...surface];
}

function buildTakeMeter(saved) {
  const model = {...saved, vocabulary:new Map(saved.words.map((word,i) => [word,i])),
    embedded:decodeFloats(saved.embedding), weights:decodeFloats(saved.coefficients)};
  return text => {
    const x = featuresFor(text, model);
    const logits = model.bias.map((b,j) => {
      for (let i = 0; i < x.length; i++) b += x[i] * model.weights[i * 3 + j];
      return b / model.temperature;
    });
    const peak = Math.max(...logits);
    const exponentials = logits.map(value => Math.exp(value - peak));
    const sum = exponentials.reduce((a,b) => a + b, 0);
    const probabilities = exponentials.map(value => value / sum);
    const index = logits.indexOf(peak);
    return {label:model.labels[index], confidence:probabilities[index],
      probabilities:Object.fromEntries(model.labels.map((label,i) => [label,probabilities[i]]))};
  };
}
if (typeof module !== 'undefined') module.exports = {buildTakeMeter, blakeFour};
