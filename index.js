const fr = require('face-recognition');
const image = fr.loadImage('img/l3m.jpg');
const detector = fr.AsyncFaceDetector();
const win = new fr.ImageWindow();
detector.locateFaces(image)
  .then((faceRectangles) => {
    win.setImage(image);
    for (var i = 0; i < faceRectangles.length; i++) {
      win.addOverlay(faceRectangles[i].rect);
    }
    fr.hitEnterToContinue();
  })
  .catch((error) => {
    console.log(error);
  });
