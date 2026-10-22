module.exports = {
  run: [{
    // app/ holds the GUI code from this repo, so reset removes only what install.js created.
    method: "fs.rm",
    params: {
      path: "app/env"
    }
  }, {
    method: "fs.rm",
    params: {
      path: "app/humaneness-vc"
    }
  }]
}
