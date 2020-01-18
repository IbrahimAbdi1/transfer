<!DOCTYPE html>
<html lang="en">
	<head>
		<meta charset="utf-8">
		<title>addit01</title>
	</head>
	<body>
		<h3>The result</h3>
		<?php
			$answer=$_REQUEST['arg1']+$_REQUEST['arg2'];
		?>
		<?php echo $_REQUEST['arg1']?> + <?php echo $_REQUEST['arg2']?> = <?php echo $answer ?>
	</body>
</html>
