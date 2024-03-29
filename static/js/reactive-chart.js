Vue.component("reactive-chart", {
  props: ["chart"],
  template: '<div :ref="chart.uuid"></div>',
  mounted() {
    Plotly.plot(this.$refs[this.chart.uuid], this.chart.traces, this.chart.layout);
  },
  watch: {
    chart: {
      handler: function () {
        Plotly.react(
          this.$refs[this.chart.uuid],
          this.chart.traces,
          this.chart.layout
        );
      },
      deep: true
    }
  }
});