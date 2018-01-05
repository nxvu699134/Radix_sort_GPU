- Tách code thành các file riêng lẻ
- Tối ưu exclusive scan
- Thay đổi cách tính histogram (1 thread sẽ tính giá trị cho 1 block trong exclusive scan)
=> mục đích để tính phần đánh hệ số cho giá trị input
=> tối ưu quá trình scatter

Time:
- Histogram: 81
- ExScan: 12
- Scatter: 0
