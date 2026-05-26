load('series.mat')
series_c = squeeze(series(16,16,:));
series_c(find(series_c == 0)) = [];
plot(histcounts(series_c,230))

hist = series2hist(series,80.33);
figure
plot(squeeze(hist(16,16,:)))
my_display_hist()