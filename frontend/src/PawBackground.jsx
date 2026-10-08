import "./PawBackground.css";

// x, y in % of screen, size in px, dir = spin direction, speed = seconds per full turn
const PAWS = [
  { x: 6,  y: 10, size: 90,  dir: "normal",  speed: 10, color: "blue" },
  { x: 82, y: 6,  size: 70,  dir: "reverse", speed: 15 , color: "yellow" },
  { x: 90, y: 42, size: 110, dir: "normal",  speed: 10, color: "blue" },
  { x: 3,  y: 55, size: 75,  dir: "reverse", speed: 15, color: "yellow" },
  { x: 45, y: 80, size: 95,  dir: "normal",  speed: 15, color: "yellow" },
  { x: 72, y: 78, size: 65,  dir: "reverse", speed: 15, color: "blue" },
  { x: 25, y: 30, size: 55,  dir: "reverse", speed: 18, color: "blue" },
];

function Paw() {
  return (
    <svg viewBox="0 0 64 64" aria-hidden="true">
      <ellipse cx="14" cy="28" rx="5" ry="7" transform="rotate(-20 14 28)" />
      <ellipse cx="25" cy="16" rx="5.5" ry="7.5" transform="rotate(-8 25 16)" />
      <ellipse cx="39" cy="16" rx="5.5" ry="7.5" transform="rotate(8 39 16)" />
      <ellipse cx="50" cy="28" rx="5" ry="7" transform="rotate(20 50 28)" />
      <path d="M32 32c-8 0-16 8-18 15c-2 7 3 11 9 10c4-1 6-2 9-2c3 0 5 1 9 2c6 1 11-3 9-10c-2-7-10-15-18-15z" />
    </svg>
  );
}

export default function PawBackground() {
  return (
    <div className="paw-bg" aria-hidden="true">
      {PAWS.map((p, i) => (
        <div
          key={i}
          className={`paw paw-${p.color}`}
          style={{
            left: `${p.x}%`,
            top: `${p.y}%`,
            width: p.size,
            height: p.size,
            animationDuration: `${p.speed}s`,
            animationDirection: p.dir,
          }}
        >
          <Paw />
        </div>
      ))}
    </div>
  );
}