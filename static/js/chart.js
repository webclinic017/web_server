Vue.component("chart", {
    props: {
        symbol: '',
        task: ''
    },
    data() {
        return {
            fig: {},
        }
    },
    methods: {
        async plot() {
            this.fig = await axios.get('/trends', {
                params: {
                    'symbol': 'TSLA'
                }
            });
            Plotly.react('plot', this.fig.data, this.fig.layout);
        },
    },
    mounted() {
        this.plot()
    },
    render(createElement) {
        return createElement('div', {
            attrs: {
                id: 'plot',
                style: "width:600px;height:250px;"
            }
        })
    },
});