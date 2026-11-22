import {
  flexRender, getCoreRowModel, getSortedRowModel, useReactTable,
  type ColumnDef, type SortingState,
} from "@tanstack/react-table";
import { useState } from "react";

/** Column options read from `meta`: numeric (right-aligned, kept on one line), hideSm (left out on a phone),
 *  width (share of the table, e.g. "24%"), className (extra class on th and td). */
type Meta = { numeric?: boolean; hideSm?: boolean; width?: string; className?: string };

/** TanStack Table with sortable headers.
 *  On narrow screens each row becomes a block of "label: value" lines (every cell carries its header as
 *  data-label), so wide tables never scroll sideways. Pass stack={false} to keep a compact table of numbers;
 *  mark its less important columns with meta.hideSm. */
export function DataTable<T>({
  data, columns, caption, highlight, stack = true, narrow = false, fixed = false,
}: {
  data: T[]; columns: ColumnDef<T, any>[]; caption: string; highlight?: (row: T) => boolean;
  stack?: boolean; narrow?: boolean; fixed?: boolean;
}) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const table = useReactTable({
    data, columns, state: { sorting }, onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(), getSortedRowModel: getSortedRowModel(),
  });
  const cls = (m: Meta | undefined) => [m?.numeric ? "n" : "", m?.hideSm ? "hide-sm" : "", m?.className ?? ""].filter(Boolean).join(" ") || undefined;
  return (
    <div className="table-wrap" data-narrow={narrow ? "true" : undefined}>
      <table className="data" data-stack={stack ? "true" : undefined} data-fixed={fixed ? "true" : undefined}>
        <caption className="sr-only">{caption}</caption>
        <thead>
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id}>
              {hg.headers.map((h) => {
                const meta = h.column.columnDef.meta as Meta | undefined;
                const sorted = h.column.getIsSorted();
                return (
                  <th key={h.id} scope="col" className={cls(meta)} style={meta?.width ? { width: meta.width } : undefined}
                    aria-sort={sorted === "asc" ? "ascending" : sorted === "desc" ? "descending" : undefined}>
                    {h.column.getCanSort() ? (
                      <button type="button" onClick={h.column.getToggleSortingHandler()} title="Sort by this column">
                        {flexRender(h.column.columnDef.header, h.getContext())}
                        <span aria-hidden="true">{sorted === "asc" ? " ▲" : sorted === "desc" ? " ▼" : ""}</span>
                      </button>
                    ) : flexRender(h.column.columnDef.header, h.getContext())}
                  </th>
                );
              })}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id} data-highlight={highlight ? highlight(row.original) : undefined}>
              {row.getVisibleCells().map((cell) => {
                const meta = cell.column.columnDef.meta as Meta | undefined;
                const head = cell.column.columnDef.header;
                return (
                  <td key={cell.id} className={cls(meta)} data-label={typeof head === "string" ? head : undefined}>
                    <div className="cell">{flexRender(cell.column.columnDef.cell, cell.getContext())}</div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {table.getRowModel().rows.length === 0 ? <p className="small table-empty">Nothing matches. Clear the search or choose another group or status.</p> : null}
    </div>
  );
}
