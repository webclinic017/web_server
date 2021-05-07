Vue.component("toolbar", {

  template: `
  <v-card
    color="grey lighten-4"
    height="50px"
    tile
  >
    <v-toolbar dense elevation="20" rounded shaped short>
      <v-app-bar-nav-icon></v-app-bar-nav-icon>

      <v-toolbar-title>Horus</v-toolbar-title>

      <v-spacer></v-spacer>

      <v-btn icon>
        <v-icon>mdi-dots-vertical</v-icon>
      </v-btn>
    </v-toolbar>
  </v-card>
`
});