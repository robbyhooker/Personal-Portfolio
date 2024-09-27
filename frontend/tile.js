class Tile {
  constructor(x, y, type, c) {
    this.x = x;
    this.y = y;
    this.type = type;
    this.c = c;
  }

  display() {
    push();
    translate(this.x, this.y);

    stroke(this.c);
    noFill();
    strokeWeight(10);
    if (this.type == 0) {
      arc(0, 0, size, size, 0, 90);
      arc(size, size, size, size, 180, 270);
    } else {
      arc(size, 0, size, size, 90, 180);
      arc(0, size, size, size, 270, 360);
    }
    pop();
  }
}
