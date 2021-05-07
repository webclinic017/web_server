Vue.component("watchlist", {
  data() {
    return {
      search: '',
      expanded: [],
      headers: [],
      data: [],
      selected: [],
      news: [],
      trends: '',
    }
  },
  methods: {
    async get_watchlist() {
      this.data = (await axios.get('/watchlist')).data;
      this.generate_headers_watchlist();
    },
    generate_headers_watchlist() {
      var keys = Object.keys(this.data[0]);
      this.headers = [];
      for (var i = 0; i < keys.length; i++) {
        this.headers.push({
          text: keys[i],
          value: keys[i]
        });
      }
      console.log(this.headers)
    },
    async get_news(symbol) {
      this.news = (await axios.get('/news', {
        params: {
          'symbol': symbol
        }
      })).data;
      console.log("News from backend : ", this.news);
    },
    async get_trends(symbol) {
      this.trends = (await axios.get('/trends', {
        params: {
          'symbol': symbol
        }
      })).data;
      console.log("Trends from backend : ", this.trends);
      //this.trends = this.to_image(this.trends);
    },
    clean() {
      this.news = []
    },
    to_image(blob) {
      return 'data:image/png;base64,' + btoa(
        new Uint8Array(blob)
        .reduce((data, byte) => data + String.fromCharCode(byte), '')
      );
    },
  },

  mounted() {
    this.get_watchlist();
    //console.log(this.$vuetify.breakpoint)
    //let recaptchaScript = document.createElement('script')
    //recaptchaScript.setAttribute('src', '/static/js/tradingview.js')
    //document.head.appendChild(recaptchaScript)
  },
  created() {
    setInterval(() => {
      this.get_watchlist()
    }, 600000) //call spre backend la 600 secunde
  },
  template: `
<div>
  <v-card>

    <v-card-title>
      Watchlist
      <v-spacer></v-spacer>
      <v-text-field
        v-model="search"
        append-icon="mdi-magnify"
        label="Search"
        single-line
        hide-details
      ></v-text-field>
    </v-card-title>

    <v-data-table
      v-model="selected"
      dense
      :headers="headers"
      :items="data"
      :sort-by="['col1', 'col2']"
      :sort-desc="[false, true]"
      :search="search"
      multi-sort
      class="elevation-1"
      hide-default-footer
      :single-expand=true
      :expanded.sync="expanded"
      :single-select=false
      item-key="symbol"
      show-select
      @click:row="clean"
      show-expand>


      <template v-slot:expanded-item="{ headers, item }" v-on:click="clean">
        <!--td :colspan="headers.length">More info about {{ item.symbol }}</td-->


        <v-row align="center" justify="center">
            <v-btn rounded>
              <v-btn text small color="primary" v-on:click="get_news(item.symbol)">News</v-btn>
       
              <v-btn text small color="primary" v-on:click="get_trends(item.symbol)">Trends</v-btn>
        
              <v-btn text small color="primary" v-on:click="get_chart(item.symbol)">Chart</v-btn>
            </v-btn>
        </v-row>


        <li v-for="elem in news">
          <pre>
          {{elem.date}}  >>  {{elem.title}} <a :href="elem.link">{{elem.link}}</a> <br/>
              {{elem.desc}} <br/>
          </pre>
        </li>

        <img src="/static/img/image.png">


      </template>
    </v-data-table>

  </v-card>
</div>
`
});