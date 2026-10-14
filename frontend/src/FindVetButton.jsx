import "./FindVetButton.css";

const DISEASES = ["Scabies", "Flea Allergy", "Ringworm"];

const TIPS = {
  Scabies:
    "Scabies is highly contagious. Try to see a vet soon and keep this cat away from other cats until it is treated.",
  Ringworm:
    "A vet can confirm ringworm with a simple test. It can spread to people, so wash your hands after handling your cat.",
  "Flea Allergy":
    "A vet can recommend a safe flea treatment for all the pets in your home.",
};

const MAPS_URL =
  "https://www.google.com/maps/search/?api=1&query=" +
  encodeURIComponent("veterinary clinic near me");

export default function FindVetButton({ result }) {
  const label = result?.prediction?.label;
  if (result?.result_type !== "disease" || !DISEASES.includes(label)) return null;

  return (
    <div className="find-vet">
      <p className="find-vet-tip">{TIPS[label]}</p>
      <a
        className="find-vet-btn"
        href={MAPS_URL}
        target="_blank"
        rel="noopener noreferrer"
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 2C8.1 2 5 5.1 5 9c0 5.2 7 13 7 13s7-7.8 7-13c0-3.9-3.1-7-7-7zm0 9.5A2.5 2.5 0 1 1 12 6.5a2.5 2.5 0 0 1 0 5z" />
        </svg>
        Find a vet near me
      </a>
      <p className="find-vet-note">
        Opens Google Maps in a new tab. FelineHealth-AI does not receive your location.
      </p>
    </div>
  );
}