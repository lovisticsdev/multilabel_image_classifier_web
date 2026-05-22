const LABEL_OVERRIDES: Record<string, string> = {
  diningtable: 'Dining table',
  motorbike: 'Motorbike',
  pottedplant: 'Potted plant',
  tvmonitor: 'TV monitor',
};

export function formatLabel(label: string): string {
  return LABEL_OVERRIDES[label] ?? label.replace(/[_-]+/g, ' ').replace(/\b\w/g, (match) => match.toUpperCase());
}

export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}
