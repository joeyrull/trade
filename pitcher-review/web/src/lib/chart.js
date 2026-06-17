// Downsample an array to at most maxPts points for chart performance, keeping
// every Nth element (cheap, preserves shape). Shared by the metrics charts.
export function downsample(arr, maxPts = 300) {
  if (arr.length <= maxPts) return arr;
  const step = Math.ceil(arr.length / maxPts);
  return arr.filter((_, i) => i % step === 0);
}
