param(
    [string]$InputPath = "data/fuel_label_dataset/all_labeled_points.csv",
    [string]$OutputDir = "artifacts/slide_assets/signal_examples",
    [int]$ContextPoints = 14
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

function As-Number($value, [double]$default = 0.0) {
    $parsed = 0.0
    if ([double]::TryParse([string]$value, [ref]$parsed)) { return $parsed }
    return $default
}

function Select-Example($groups, [scriptblock]$predicate, [scriptblock]$sortExpression, [bool]$descending) {
    $candidates = foreach ($group in $groups) {
        $points = @($group.Group | Sort-Object FuelTime)
        for ($index = 3; $index -lt ($points.Count - 3); $index++) {
            $point = $points[$index]
            if (& $predicate $point) {
                [pscustomobject]@{ Point = $point; Index = $index; Points = $points }
            }
        }
    }

    if (-not $candidates) { throw "Khong tim thay mau phu hop." }
    if ($descending) {
        return $candidates | Sort-Object $sortExpression -Descending | Select-Object -First 1
    }
    return $candidates | Sort-Object $sortExpression | Select-Object -First 1
}

function Draw-Example($example, [string]$fileName, [string]$title, [string]$caption, [System.Drawing.Color]$highlight) {
    $width, $height = 1400, 720
    $left, $right, $top, $bottom = 115, 70, 105, 105
    $plotWidth, $plotHeight = $width - $left - $right, $height - $top - $bottom

    $start = [Math]::Max(0, $example.Index - $ContextPoints)
    $end = [Math]::Min($example.Points.Count - 1, $example.Index + $ContextPoints)
    $window = @($example.Points[$start..$end])
    $values = @($window | ForEach-Object { As-Number $_.FuelLevel })
    $minValue = ($values | Measure-Object -Minimum).Minimum
    $maxValue = ($values | Measure-Object -Maximum).Maximum
    $padding = [Math]::Max(($maxValue - $minValue) * 0.16, 0.5)
    $minValue -= $padding
    $maxValue += $padding
    if ($maxValue -eq $minValue) { $maxValue += 1.0 }

    $bitmap = [System.Drawing.Bitmap]::new($width, $height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.Clear([System.Drawing.Color]::White)

    $titleFont = [System.Drawing.Font]::new("Arial", 28, [System.Drawing.FontStyle]::Bold)
    $textFont = [System.Drawing.Font]::new("Arial", 16)
    $axisFont = [System.Drawing.Font]::new("Arial", 14)
    $gridPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(220, 226, 235), 1)
    $linePen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(225, 42, 42), 3)
    $markerBrush = [System.Drawing.SolidBrush]::new($highlight)
    $markerPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(65, 65, 65), 2)
    $textBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(28, 43, 65))
    $mutedBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(96, 110, 130))

    $graphics.DrawString($title, $titleFont, $textBrush, 52, 28)
    $graphics.DrawString($caption, $textFont, $mutedBrush, 54, 70)

    for ($tick = 0; $tick -le 5; $tick++) {
        $y = $top + $plotHeight * $tick / 5.0
        $graphics.DrawLine($gridPen, $left, $y, $width - $right, $y)
        $labelValue = $maxValue - ($maxValue - $minValue) * $tick / 5.0
        $label = $labelValue.ToString("0.0")
        $graphics.DrawString($label, $axisFont, $mutedBrush, 34, $y - 10)
    }
    $graphics.DrawString("Muc nhien lieu (L)", $axisFont, $mutedBrush, 52, $height - 54)

    $points = [System.Collections.Generic.List[System.Drawing.PointF]]::new()
    for ($i = 0; $i -lt $window.Count; $i++) {
        $x = $left + $plotWidth * $i / [Math]::Max($window.Count - 1, 1)
        $y = $top + $plotHeight * ($maxValue - $values[$i]) / ($maxValue - $minValue)
        $points.Add([System.Drawing.PointF]::new([float]$x, [float]$y))
    }
    if ($points.Count -gt 1) { $graphics.DrawLines($linePen, $points.ToArray()) }
    foreach ($point in $points) { $graphics.FillEllipse([System.Drawing.Brushes]::Red, $point.X - 3, $point.Y - 3, 6, 6) }

    $eventIndex = $example.Index - $start
    $eventPoint = $points[$eventIndex]
    $graphics.FillEllipse($markerBrush, $eventPoint.X - 10, $eventPoint.Y - 10, 20, 20)
    $graphics.DrawEllipse($markerPen, $eventPoint.X - 10, $eventPoint.Y - 10, 20, 20)
    $graphics.DrawString("Diem mau", $textFont, $textBrush, $eventPoint.X + 14, $eventPoint.Y - 33)

    $timeText = [string]$example.Point.FuelTime
    $detail = "Xe: $($example.Point.VehicleID) | Thoi gian: $timeText | Fuel: $(([Math]::Round((As-Number $example.Point.FuelLevel), 1))) L"
    $graphics.DrawString($detail, $axisFont, $mutedBrush, $left, $height - 30)

    $pngPath = Join-Path $OutputDir $fileName
    $bitmap.Save($pngPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose(); $bitmap.Dispose()
    $titleFont.Dispose(); $textFont.Dispose(); $axisFont.Dispose(); $gridPen.Dispose(); $linePen.Dispose(); $markerBrush.Dispose(); $markerPen.Dispose(); $textBrush.Dispose(); $mutedBrush.Dispose()

    return [pscustomobject]@{
        File = $pngPath
        Title = $title
        VehicleID = $example.Point.VehicleID
        FuelTime = $example.Point.FuelTime
        FuelLevel = [Math]::Round((As-Number $example.Point.FuelLevel), 2)
        Label = $example.Point.Label
    }
}

if (-not (Test-Path $InputPath)) { throw "Khong tim thay file nhan: $InputPath" }
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$wantedLabels = @("STABLE_JITTER", "CONSUMPTION", "REFUEL", "DRAIN", "SLOSHING_NOISE")
$rows = @(Import-Csv -Path $InputPath | Where-Object { $_.Label -in $wantedLabels })
$groups = @($rows | Group-Object Source, VehicleID)

$stable = Select-Example $groups { param($p) $p.Label -eq "STABLE_JITTER" -and (As-Number $p.RollingStd12) -le 0.25 -and [Math]::Abs((As-Number $p.DeltaFuel)) -le 0.25 } { param($c) As-Number $c.Point.RollingStd12 } $false
$consumption = Select-Example $groups { param($p) $p.Label -eq "CONSUMPTION" -and (As-Number $p.DeltaFuel) -lt 0 -and (As-Number $p.RollingStd12) -lt 2.0 } { param($c) As-Number $c.Point.RollingStd12 } $false
$refuel = Select-Example $groups { param($p) $p.Label -eq "REFUEL" -and (As-Number $p.DeltaFuel) -gt 0 } { param($c) As-Number $c.Point.DeltaFuel } $true
$drain = Select-Example $groups { param($p) $p.Label -eq "DRAIN" -and (As-Number $p.DeltaFuel) -lt 0 } { param($c) As-Number $c.Point.DeltaFuel } $false
$sloshing = Select-Example $groups { param($p) $p.Label -eq "SLOSHING_NOISE" -and (As-Number $p.RollingStd12) -gt 0 } { param($c) As-Number $c.Point.RollingStd12 } $true
$spike = Select-Example $groups { param($p) $p.Label -eq "SLOSHING_NOISE" -and ((As-Number $p.PeakReversalFlag) -eq 1 -or (As-Number $p.ValleyReversalFlag) -eq 1) } { param($c) As-Number $c.Point.AbsDeltaFuel } $true

$manifest = @()
$manifest += Draw-Example $refuel "01_refuel.png" "REFUEL - Tang nhien lieu that" "Tang lon duy tri sau su kien, can bam nhanh." ([System.Drawing.Color]::FromArgb(21, 159, 119))
$manifest += Draw-Example $drain "02_drain.png" "DRAIN - Giam nhien lieu bat thuong" "Giam lon can duoc phan biet voi nhieu ngan han." ([System.Drawing.Color]::FromArgb(213, 53, 53))
$manifest += Draw-Example $consumption "03_consumption.png" "CONSUMPTION - Giam nhien lieu theo xu huong" "Nhien lieu giam tu tu va duy tri cung chieu theo thoi gian." ([System.Drawing.Color]::FromArgb(41, 114, 181))
$manifest += Draw-Example $stable "04_stable_jitter.png" "STABLE_JITTER - Dao dong nho quanh muc on dinh" "Dao dong nho can duoc giu phang bang deadband." ([System.Drawing.Color]::FromArgb(45, 156, 68))
$manifest += Draw-Example $sloshing "05_sloshing_noise.png" "SLOSHING_NOISE - Dao dong manh do rung lac/song sanh" "Nhieu lien tuc khong nen duoc bo loc bam theo tung diem raw." ([System.Drawing.Color]::FromArgb(238, 131, 24))
$manifest += Draw-Example $spike "06_dropout_spike.png" "DROPOUT/SPIKE - Nhay/tut dot ngot roi quay lai" "Su kien ngan han can duoc tu choi thay vi cap nhat muc nhien lieu." ([System.Drawing.Color]::FromArgb(108, 79, 169))

$manifest | Export-Csv -Path (Join-Path $OutputDir "signal_examples_manifest.csv") -NoTypeInformation -Encoding utf8
Write-Host "Da xuat $($manifest.Count) anh vao: $((Resolve-Path $OutputDir).Path)"
$manifest | Format-Table -AutoSize
