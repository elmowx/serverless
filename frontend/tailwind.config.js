/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Fraunces",
          "ui-serif",
          "Georgia",
          '"Times New Roman"',
          "serif",
        ],
        serif: [
          "Fraunces",
          "ui-serif",
          "Georgia",
          '"Times New Roman"',
          "serif",
        ],
      },
      colors: {
        paper: "#f6f2eb",
        ink: "#1f1b16",
        muted: "#8a8278",
        accent: "#ff8a5b",
        warm: "#fbe4c9",
      },
    },
  },
  plugins: [],
};
