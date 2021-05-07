Vue.component("date_picker", {
  data: () => ({
    dates: [],
    menu: false,
  }),
  computed: {
    dateRangeText() {
      return this.dates.join(' ~ ')
    },
  },
  created() {
    var currentDate = new Date().toJSON().slice(0, 10).replace(/-/g, '-');
    this.dates = [currentDate, currentDate];
  },
  template: `
<div>
  <v-menu
    ref="menu"
    v-model="menu"
    :close-on-content-click="false"
    transition="scale-transition"
    offset-y
    min-width="290px">

    <template v-slot:activator="{ on, attrs }">
      <v-text-field
        v-model="dateRangeText"
        label="Date range"
        readonly
        v-bind="attrs"
        v-on="on"
      ></v-text-field>
    </template>

    <v-date-picker 
        v-model="dates" 
        range
        persistent-hint
        no-title>
        <v-btn text color="primary" @click="menu = false">Cancel</v-btn>
        <v-btn text color="primary" @click="$refs.menu.save(dates)">OK</v-btn>
    </v-date-picker>
  </v-menu>
</div>
`
});