import type { ControlCategory } from "../types";

interface CategoryColumn {
  category: ControlCategory;
  label: string;
  methods: string[];
}

const COLUMNS: CategoryColumn[] = [
  { category: "input_control", label: "Input", methods: ["few_shot"] },
  { category: "structural_control", label: "Structural", methods: [] },
  { category: "state_control", label: "State", methods: ["pasta", "cast", "caa", "iti", "act_add"] },
  {
    category: "output_control",
    label: "Output",
    methods: ["deal", "rad", "sasa", "thinking_intervention"],
  },
];

export function LibraryPanel() {
  return (
    <div className="library-row" role="region" aria-label="Steering controls library">
      {COLUMNS.map((col) => (
        <div className="library-column" data-category={col.category} key={col.category}>
          <div className="library-header">
            <span className="library-dot" />
            <span className="library-label">{col.label}</span>
          </div>
          <div className="library-body">
            {col.methods.length === 0 ? (
              <div className="library-empty">no methods registered</div>
            ) : (
              col.methods.map((method) => (
                <div className="library-pill" key={method} title={method}>
                  {method}
                </div>
              ))
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
