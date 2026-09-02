param(
    [string]$OutputDir = "TestDoDoc\results_new_raw_v3"
)

Add-Type -AssemblyName System.Drawing

$labels = @(
    "UPWARD_SHIFT",
    "DOWNWARD_SHIFT",
    "GRADUAL_CHANGE",
    "STABLE_JITTER",
    "OSCILLATION_NOISE"
)

Get-ChildItem -Path $OutputDir -Filter "*_state_transition_matrix.csv" | ForEach-Object {
    $rows = Import-Csv $_.FullName
    $maxValue = 1
    foreach ($row in $rows) {
        foreach ($label in $labels) {
            $maxValue = [Math]::Max($maxValue, [int]$row.$label)
        }
    }

    $bitmap = [System.Drawing.Bitmap]::new(1940, 1660)
    $g = [System.Drawing.Graphics]::FromImage($bitmap)
    $g.Clear([System.Drawing.Color]::White)
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAlias
    $titleFont = [System.Drawing.Font]::new("Arial", 26, [System.Drawing.FontStyle]::Bold)
    $labelFont = [System.Drawing.Font]::new("Arial", 16)
    $valueFont = [System.Drawing.Font]::new("Arial", 18, [System.Drawing.FontStyle]::Bold)
    $textBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(25, 35, 55))

    $base = $_.BaseName -replace "_state_transition_matrix$", ""
    $g.DrawString("$base - RF State Transition Matrix", $titleFont, $textBrush, 360, 28)
    $g.DrawString("Previous RF label", $labelFont, $textBrush, 28, 510)
    $g.DrawString("Current RF predicted label", $labelFont, $textBrush, 960, 1510)

    $left = 520; $top = 190; $cell = 260
    for ($col = 0; $col -lt $labels.Count; $col++) {
        $format = [System.Drawing.StringFormat]::new()
        $format.Alignment = [System.Drawing.StringAlignment]::Center
        $g.DrawString($labels[$col], $labelFont, $textBrush, [System.Drawing.RectangleF]::new($left + $col * $cell, 120, $cell - 4, 60), $format)
    }
    for ($rowIndex = 0; $rowIndex -lt $labels.Count; $rowIndex++) {
        $format = [System.Drawing.StringFormat]::new()
        $format.Alignment = [System.Drawing.StringAlignment]::Far
        $format.LineAlignment = [System.Drawing.StringAlignment]::Center
        $g.DrawString($labels[$rowIndex], $labelFont, $textBrush, [System.Drawing.RectangleF]::new(35, $top + $rowIndex * $cell, $left - 30, $cell - 4), $format)
        $source = $rows[$rowIndex]
        for ($col = 0; $col -lt $labels.Count; $col++) {
            $value = [int]$source.($labels[$col])
            $ratio = [double]$value / $maxValue
            $fill = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb([int](247 - 200*$ratio), [int](250 - 105*$ratio), [int](255 - 35*$ratio)))
            $x = $left + $col * $cell; $y = $top + $rowIndex * $cell
            $g.FillRectangle($fill, $x, $y, $cell - 4, $cell - 4)
            $g.DrawRectangle([System.Drawing.Pens]::LightGray, $x, $y, $cell - 4, $cell - 4)
            $center = [System.Drawing.StringFormat]::new()
            $center.Alignment = [System.Drawing.StringAlignment]::Center
            $center.LineAlignment = [System.Drawing.StringAlignment]::Center
            $valueBrush = if ($ratio -gt 0.60) { [System.Drawing.Brushes]::White } else { $textBrush }
            $g.DrawString("$value", $valueFont, $valueBrush, [System.Drawing.RectangleF]::new($x, $y, $cell - 4, $cell - 4), $center)
        }
    }
    $noteFont = [System.Drawing.Font]::new("Arial", 13, [System.Drawing.FontStyle]::Italic)
    $noteBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::DimGray)
    $g.DrawString("Raw trace has no ground-truth labels; this is a transition matrix, not an accuracy confusion matrix.", $noteFont, $noteBrush, 35, 1550)

    $png = Join-Path $_.DirectoryName ($base + "_state_transition_matrix.png")
    $bitmap.Save($png, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bitmap.Dispose()
    Write-Output "Saved $png"
}
