Vue.component("foot", {
  methods: {
    to_linkedin() {
      window.location.href = "https://www.linkedin.com/in/flaviu-marinescu-26878aa9/"
    }
  },
  template: `
<v-footer
  :padless="true"
  :absolute="false"
  :fixed="true"
>
<v-card
  class="flex"
  height="30"
  flat
  tile
  rounded
>

  <v-card-text class="py-0 text-center">
    {{ new Date().getFullYear() }} — <strong>Flaviu G. Marinescu</strong>
    <v-btn
      v-on:click="to_linkedin"
      class="mx-4"
      icon
    >
      <v-icon size="22px">mdi-linkedin</v-icon>
    </v-btn>
  </v-card-text>
</v-card>
</v-footer>
`
});