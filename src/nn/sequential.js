/**
 * Sequential：把层串成前向流水线，并汇总所有可训练参数。
 * 这是让"训练循环"能写得干净的关键——优化器只需要遍历 parameters()。
 */
export class Sequential {
  constructor(layers = []) {
    this.layers = layers;
  }

  add(layer) {
    this.layers.push(layer);
    return this;
  }

  forward(x) {
    let out = x;
    for (const layer of this.layers) out = layer.forward(out);
    return out;
  }

  /** 展平所有层的参数，方便统一更新与保存 */
  parameters() {
    const params = [];
    for (const layer of this.layers) {
      if (typeof layer.parameters === "function") params.push(...layer.parameters());
    }
    return params;
  }

  /** 参数量统计，README 里的模型规模就靠它 */
  countParams() {
    return this.parameters().reduce((sum, t) => sum + t.size, 0);
  }

  summary() {
    const rows = this.layers.map((l) => `${l.name}`);
    return `Sequential(${rows.length} 层: ${rows.join(" → ")}) | 参数量 ${this.countParams()}`;
  }

  toJSON() {
    return {
      format: "axon-model",
      version: 1,
      type: "Sequential",
      layers: this.layers.map((layer, index) => {
        if (typeof layer?.toJSON !== "function") {
          throw new Error(
            `Sequential.toJSON: layer ${index} (${layer?.name || "unnamed"}) does not implement toJSON()`
          );
        }
        return layer.toJSON();
      }),
    };
  }
}
